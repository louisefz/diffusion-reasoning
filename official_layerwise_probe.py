#!/usr/bin/env python
"""Locate answer emergence across the 12 ELF-B denoiser blocks."""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent
OFFICIAL_SRC = ROOT / "official-elf" / "src"
sys.path.insert(0, str(OFFICIAL_SRC))

from configs.config import load_config_from_yaml
from modules.t5_encoder import get_encoder
from official_diagnostics import load_model
from official_flow_trajectory import answer_decoder_stats
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split
from utils.encoder_utils import encode_text
from utils.generation_utils import _dlm_decode_logits
from utils.sampling_utils import restore_cond


GROUPS = (("compose", 1), ("compose", 2), ("compose", 4), ("lookup", 4))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-per-group", type=int, default=2048)
    parser.add_argument("--eval-per-group", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--alpha", action="append", type=float, default=None)
    return parser.parse_args()


def select_balanced(dataset, count_per_group, seed):
    """Select random examples using Arrow masks without scanning Python rows."""
    table = dataset.data.table
    rng = random.Random(seed)
    selected = []
    groups = []
    for task, depth in GROUPS:
        mask = pc.and_(
            pc.equal(table.column("task"), pa.scalar(task)),
            pc.equal(table.column("depth"), pa.scalar(depth)),
        )
        candidates = pc.indices_nonzero(mask).to_numpy().tolist()
        if len(candidates) < count_per_group:
            raise ValueError(
                f"Requested {count_per_group} for {task}/d{depth}; "
                f"only {len(candidates)} available"
            )
        chosen = sorted(rng.sample(candidates, count_per_group))
        selected.extend(chosen)
        groups.extend([f"{task}/d{depth}"] * count_per_group)
    return dataset.select(selected), groups, selected


class HiddenRecorder:
    """Capture answer-slot representations before and after every ELF block."""

    def __init__(self, model, answer_positions):
        self.model = model
        self.answer_positions = answer_positions
        self.hidden = []
        self.handles = []
        prefix = (
            model.num_model_mode_tokens + model.num_time_tokens
            + model.num_self_cond_cfg_tokens
        )

        def text_hook(_module, _inputs, output):
            rows = torch.arange(output.shape[0], device=output.device)
            self.hidden.append(output[rows, self.answer_positions].detach())

        self.handles.append(model.text_proj.register_forward_hook(text_hook))
        for block in model.blocks:
            def block_hook(_module, _inputs, output, prefix=prefix):
                rows = torch.arange(output.shape[0], device=output.device)
                positions = self.answer_positions + prefix
                self.hidden.append(output[rows, positions].detach())
            self.handles.append(block.register_forward_hook(block_hook))

    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def extract_features(dataset, expected_groups, model, encoder, tokenizer, config,
                     device, batch_size, seed):
    loader = get_dataloader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        drop_last=False, max_seq_length=config.max_length,
        pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
        max_input_seq_length=config.max_input_length, distributed=False,
    )
    dtype = next(model.parameters()).dtype
    noise_generator = torch.Generator(device="cpu").manual_seed(seed)
    feature_batches, labels, groups = [], [], []
    endpoint_correct = []
    cursor = 0
    use_bf16 = bool(config.use_bf16) and device.type == "cuda"

    for batch in loader:
        bsz = len(batch["target"])
        input_ids = torch.from_numpy(np.asarray(batch["input_ids"])).to(device).long()
        encoder_mask = torch.from_numpy(
            np.asarray(batch["encoder_attention_mask"])
        ).to(device).float()
        cond_mask = torch.from_numpy(np.asarray(batch["cond_seq_mask"])).to(device).float()
        answer_starts = cond_mask.to(torch.long).sum(dim=1)
        answer_lengths = torch.tensor(
            [len(tokenizer.encode(str(target), add_special_tokens=False))
             for target in batch["target"]], device=device, dtype=torch.long,
        )
        answer_positions = answer_starts + answer_lengths - 1
        cond_seq = encode_text(
            input_ids, encoder_mask, encoder, config.latent_mean,
            config.latent_std, use_bf16=bool(config.use_bf16),
        ).to(dtype)
        z = (torch.randn(
            (bsz, config.max_length, model.text_encoder_dim),
            generator=noise_generator, dtype=dtype,
        ) * config.denoiser_noise_scale).to(device)
        z = restore_cond(z, cond_seq, cond_mask)
        xpred_previous = restore_cond(torch.zeros_like(z), cond_seq, cond_mask)
        model_input = torch.cat([z, xpred_previous], dim=-1)
        t_batch = torch.zeros((bsz,), device=device, dtype=dtype)
        sc_batch = torch.ones((bsz,), device=device, dtype=dtype)

        recorder = HiddenRecorder(model, answer_positions)
        try:
            with torch.amp.autocast(
                "cuda", dtype=torch.bfloat16, enabled=use_bf16,
            ):
                endpoint, _ = model(
                    model_input, t_batch, deterministic=True,
                    self_cond_cfg_scale=sc_batch, decoder_step_active=None,
                )
        finally:
            recorder.close()
        if len(recorder.hidden) != model.depth + 1:
            raise RuntimeError(
                f"Expected {model.depth + 1} layers, captured {len(recorder.hidden)}"
            )
        feature_batches.append(
            torch.stack(recorder.hidden, dim=1).to("cpu", dtype=torch.bfloat16)
        )
        endpoint_logits = _dlm_decode_logits(
            endpoint, model, 1.0, config, 1.0,
        )
        endpoint_stats = answer_decoder_stats(
            endpoint_logits, answer_starts, answer_lengths, input_ids,
        )
        endpoint_correct.extend(endpoint_stats["correct"].cpu().tolist())
        labels.extend(int(target) for target in batch["target"])
        groups.extend(expected_groups[cursor:cursor + bsz])
        cursor += bsz
        print(f"Extracted {cursor}/{len(dataset)}", flush=True)

    return {
        "features": torch.cat(feature_batches).float().numpy(),
        "labels": np.asarray(labels, dtype=np.int64),
        "groups": np.asarray(groups),
        "endpoint_correct": np.asarray(endpoint_correct, dtype=bool),
    }


def train_probes(train, evaluation, alphas, seed):
    strata = np.asarray([
        f"{group}:{label}" for group, label in zip(train["groups"], train["labels"])
    ])
    fit_indices, dev_indices = train_test_split(
        np.arange(len(strata)), test_size=0.2, random_state=seed,
        stratify=strata,
    )
    metrics, probes = [], []
    layer_count = train["features"].shape[1]
    for layer in range(layer_count):
        x_train = train["features"][:, layer]
        best = None
        for alpha in alphas:
            probe = make_pipeline(
                StandardScaler(), RidgeClassifier(alpha=alpha),
            )
            probe.fit(x_train[fit_indices], train["labels"][fit_indices])
            dev_accuracy = accuracy_score(
                train["labels"][dev_indices], probe.predict(x_train[dev_indices]),
            )
            candidate = (dev_accuracy, -alpha, alpha)
            if best is None or candidate > best:
                best = candidate
        best_alpha = best[2]
        probe = make_pipeline(
            StandardScaler(), RidgeClassifier(alpha=best_alpha),
        )
        probe.fit(x_train, train["labels"])
        probes.append(probe)
        eval_predictions = probe.predict(evaluation["features"][:, layer])
        train_predictions = probe.predict(x_train)
        row = {
            "layer": layer,
            "layer_name": "input_projection" if layer == 0 else f"block_{layer}",
            "alpha": best_alpha,
            "dev_accuracy": best[0],
            "train_accuracy": accuracy_score(train["labels"], train_predictions),
            "eval_accuracy": accuracy_score(evaluation["labels"], eval_predictions),
        }
        for group in sorted(set(evaluation["groups"])):
            mask = evaluation["groups"] == group
            row[f"eval_{group.replace('/', '_')}"] = accuracy_score(
                evaluation["labels"][mask], eval_predictions[mask],
            )
        metrics.append(row)
        print(
            f"Layer {layer:02d}: eval={row['eval_accuracy']:.4f}, "
            f"alpha={best_alpha:g}", flush=True,
        )
    return metrics, probes


def _first_layer(metrics, key, threshold):
    for row in metrics:
        if row[key] >= threshold:
            return row["layer"]
    return None


def _plot(metrics, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [
        ("eval_compose_d1", "compose/d1"),
        ("eval_compose_d2", "compose/d2"),
        ("eval_compose_d4", "compose/d4"),
        ("eval_lookup_d4", "lookup/d4"),
    ]
    fig, axis = plt.subplots(figsize=(8, 5))
    layers = [row["layer"] for row in metrics]
    for key, label in keys:
        axis.plot(layers, [row[key] for row in metrics], marker="o", label=label)
    axis.axhline(0.125, color="black", linestyle=":", linewidth=1, label="chance")
    axis.axhline(0.9, color="gray", linestyle="--", linewidth=1)
    axis.set_xlabel("ELF-B representation (0=input projection, 1–12=blocks)")
    axis.set_ylabel("Held-out linear-probe accuracy")
    axis.set_ylim(0, 1.03)
    axis.set_xticks(layers)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "layerwise_probe_accuracy.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config_from_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name or config.encoder_model_name
    )
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    model, checkpoint_step = load_model(
        config, args.checkpoint, encoder_config, tokenizer, device,
    )
    train_dataset = load_dataset_split(config.data_path)
    eval_dataset = load_dataset_split(config.eval_data_path)
    train_selected, train_groups, train_ids = select_balanced(
        train_dataset, args.train_per_group, args.seed,
    )
    eval_selected, eval_groups, eval_ids = select_balanced(
        eval_dataset, args.eval_per_group, args.seed + 1,
    )
    train = extract_features(
        train_selected, train_groups, model, encoder, tokenizer, config,
        device, args.batch_size, args.seed + 2,
    )
    evaluation = extract_features(
        eval_selected, eval_groups, model, encoder, tokenizer, config,
        device, args.batch_size, args.seed + 3,
    )
    alphas = args.alpha or [0.1, 1.0, 10.0, 100.0]
    metrics, probes = train_probes(train, evaluation, alphas, args.seed)
    endpoint_by_group = {
        group: float(evaluation["endpoint_correct"][evaluation["groups"] == group].mean())
        for group in sorted(set(evaluation["groups"]))
    }
    emergence = {
        group: {
            "accuracy_50": _first_layer(
                metrics, f"eval_{group.replace('/', '_')}", 0.5,
            ),
            "accuracy_80": _first_layer(
                metrics, f"eval_{group.replace('/', '_')}", 0.8,
            ),
            "accuracy_90": _first_layer(
                metrics, f"eval_{group.replace('/', '_')}", 0.9,
            ),
        }
        for group in endpoint_by_group
    }
    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": checkpoint_step,
        "device": str(device),
        "seed": args.seed,
        "train_per_group": args.train_per_group,
        "eval_per_group": args.eval_per_group,
        "train_source_ids": train_ids,
        "eval_source_ids": eval_ids,
        "alphas": alphas,
        "layers": model.depth,
        "hidden_size": model.hidden_size,
        "analysis_point": "conditional denoiser forward at t=0 with zero previous endpoint",
        "conditional_endpoint_accuracy": endpoint_by_group,
        "emergence_layers": emergence,
        "metrics": metrics,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    with (output_dir / "probe_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    joblib.dump(probes, output_dir / "ridge_probes.joblib", compress=3)
    torch.save({
        "eval_features": torch.from_numpy(evaluation["features"]).to(torch.bfloat16),
        "eval_labels": torch.from_numpy(evaluation["labels"]),
        "eval_groups": evaluation["groups"].tolist(),
    }, output_dir / "eval_features.pt")
    _plot(metrics, output_dir)
    print(json.dumps({
        "conditional_endpoint_accuracy": endpoint_by_group,
        "emergence_layers": emergence,
    }, indent=2), flush=True)
    print(f"Saved layerwise artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
