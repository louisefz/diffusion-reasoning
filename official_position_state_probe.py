#!/usr/bin/env python
"""Map composition states across semantic token roles and ELF blocks."""

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "official-elf" / "src"))

from configs.config import load_config_from_yaml
from modules.t5_encoder import get_encoder
from official_diagnostics import load_model
from official_intermediate_state_probe import select_compose_d4
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split
from utils.encoder_utils import encode_text
from utils.sampling_utils import restore_cond


ROLE_NAMES = (
    "start",
    "program_1", "program_2", "program_3", "program_4",
    "oracle_cell_1", "oracle_cell_2", "oracle_cell_3", "oracle_cell_4",
    "answer",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--eval-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--alpha", action="append", type=float, default=None)
    return parser.parse_args()


def overlapping_tokens(offsets, span):
    start, end = span
    result = [index for index, (left, right) in enumerate(offsets)
              if right > left and left < end and right > start]
    if not result:
        raise ValueError(f"No tokens overlap character span {span}")
    return result


def semantic_role_indices(row, tokenizer):
    """Return condition-token indices for aligned roles; answer is added later.

    oracle_cell_k is deliberately selected using the ground-truth execution
    path. It is an input/lexical anchor, not evidence of latent computation.
    """
    prompt = str(row["input"])
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    if list(encoded["input_ids"]) != list(map(int, row["condition_input_ids"])):
        raise ValueError("Retokenized prompt does not match stored condition_input_ids")

    tables = {}
    for match in re.finditer(r"F([0-3]): ([0-7](?: [0-7]){7})", prompt):
        fn = int(match.group(1))
        values = list(re.finditer(r"[0-7]", match.group(2)))
        if len(values) != 8:
            raise ValueError("Expected eight entries per function table")
        base = match.start(2)
        for input_value, value_match in enumerate(values):
            tables[(fn, input_value)] = (
                base + value_match.start(), base + value_match.end(),
            )
    start_match = re.search(r"Start ([0-7])\.", prompt)
    program_match = re.search(r"Program (F[0-3](?: F[0-3]){3})\.", prompt)
    if len(tables) != 32 or start_match is None or program_match is None:
        raise ValueError(f"Could not parse prompt: {prompt}")
    program_spans = []
    for match in re.finditer(r"F[0-3]", program_match.group(1)):
        base = program_match.start(1)
        program_spans.append((base + match.start(), base + match.end()))

    program = list(map(int, row["program"]))
    states = list(map(int, row["states"]))
    if len(program) != 4 or len(states) != 5:
        raise ValueError("Expected d4 program and [start,s1,s2,s3,s4]")
    roles = {
        "start": overlapping_tokens(offsets, start_match.span(1)),
    }
    for step in range(4):
        roles[f"program_{step + 1}"] = overlapping_tokens(offsets, program_spans[step])
        roles[f"oracle_cell_{step + 1}"] = overlapping_tokens(
            offsets, tables[(program[step], states[step])],
        )
    return roles


class RoleRecorder:
    def __init__(self, model, role_mask):
        self.hidden = []
        self.handles = []
        self.role_mask = role_mask
        self.prefix = (model.num_model_mode_tokens + model.num_time_tokens
                       + model.num_self_cond_cfg_tokens)

        def text_hook(_module, _inputs, output):
            self.hidden.append(torch.einsum("brs,bsh->brh", self.role_mask, output).detach())

        def block_hook(_module, _inputs, output):
            sequence = output[:, self.prefix:self.prefix + self.role_mask.shape[-1]]
            self.hidden.append(torch.einsum("brs,bsh->brh", self.role_mask, sequence).detach())

        self.handles.append(model.text_proj.register_forward_hook(text_hook))
        for block in model.blocks:
            self.handles.append(block.register_forward_hook(block_hook))

    def close(self):
        for handle in self.handles:
            handle.remove()


def build_role_mask(rows, tokenizer, cond_lengths, answer_lengths, sequence_length, device):
    mask = torch.zeros((len(rows), len(ROLE_NAMES), sequence_length), device=device)
    for batch_index, row in enumerate(rows):
        indices = semantic_role_indices(row, tokenizer)
        indices["answer"] = [int(cond_lengths[batch_index] + answer_lengths[batch_index] - 1)]
        for role_index, role in enumerate(ROLE_NAMES):
            positions = indices[role]
            weight = 1.0 / len(positions)
            mask[batch_index, role_index, positions] = weight
    return mask


@torch.no_grad()
def extract(dataset, model, encoder, tokenizer, config, device, batch_size, seed):
    loader = get_dataloader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        drop_last=False, max_seq_length=config.max_length,
        pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
        max_input_seq_length=config.max_input_length, distributed=False,
    )
    dtype = next(model.parameters()).dtype
    generator = torch.Generator(device="cpu").manual_seed(seed)
    feature_batches = []
    cursor = 0
    use_bf16 = bool(config.use_bf16) and device.type == "cuda"
    for batch in loader:
        bsz = len(batch["target"])
        rows = [dataset[index] for index in range(cursor, cursor + bsz)]
        input_ids = torch.from_numpy(np.asarray(batch["input_ids"])).to(device).long()
        encoder_mask = torch.from_numpy(np.asarray(batch["encoder_attention_mask"])).to(device).float()
        cond_mask = torch.from_numpy(np.asarray(batch["cond_seq_mask"])).to(device).float()
        cond_lengths = cond_mask.long().sum(dim=1)
        answer_lengths = torch.tensor(
            [len(tokenizer.encode(str(target), add_special_tokens=False)) for target in batch["target"]],
            device=device, dtype=torch.long,
        )
        role_mask = build_role_mask(
            rows, tokenizer, cond_lengths, answer_lengths, config.max_length, device,
        )
        cond_seq = encode_text(input_ids, encoder_mask, encoder, config.latent_mean,
                               config.latent_std, use_bf16=bool(config.use_bf16)).to(dtype)
        z = (torch.randn((bsz, config.max_length, model.text_encoder_dim),
                         generator=generator, dtype=dtype)
             * config.denoiser_noise_scale).to(device)
        z = restore_cond(z, cond_seq, cond_mask)
        previous = restore_cond(torch.zeros_like(z), cond_seq, cond_mask)
        recorder = RoleRecorder(model, role_mask)
        try:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                model(torch.cat([z, previous], dim=-1),
                      torch.zeros((bsz,), device=device, dtype=dtype),
                      deterministic=True,
                      self_cond_cfg_scale=torch.ones((bsz,), device=device, dtype=dtype),
                      decoder_step_active=None)
        finally:
            recorder.close()
        if len(recorder.hidden) != model.depth + 1:
            raise RuntimeError(f"Captured {len(recorder.hidden)} representations")
        feature_batches.append(torch.stack(recorder.hidden, dim=1).cpu().to(torch.bfloat16))
        cursor += bsz
        print(f"Extracted {cursor}/{len(dataset)}", flush=True)
    return torch.cat(feature_batches).float().numpy()


def fit_probes(train_features, train_states, eval_features, eval_states, alphas, seed):
    rows = []
    for step in range(1, 5):
        labels = train_states[:, step]
        fit, dev = train_test_split(
            np.arange(len(labels)), test_size=0.2, random_state=seed + step,
            stratify=labels,
        )
        for role_index, role in enumerate(ROLE_NAMES):
            for layer in range(train_features.shape[1]):
                x = train_features[:, layer, role_index]
                best = None
                for alpha in alphas:
                    probe = make_pipeline(StandardScaler(), RidgeClassifier(alpha=alpha))
                    probe.fit(x[fit], labels[fit])
                    score = accuracy_score(labels[dev], probe.predict(x[dev]))
                    candidate = (score, -alpha, alpha)
                    if best is None or candidate > best:
                        best = candidate
                probe = make_pipeline(StandardScaler(), RidgeClassifier(alpha=best[2]))
                probe.fit(x, labels)
                accuracy = accuracy_score(
                    eval_states[:, step], probe.predict(eval_features[:, layer, role_index]),
                )
                rows.append({
                    "state_step": step, "role": role, "layer": layer,
                    "alpha": best[2], "dev_accuracy": best[0],
                    "eval_accuracy": accuracy,
                    "oracle_input_anchor": role.startswith("oracle_cell_"),
                })
        top = sorted((r for r in rows if r["state_step"] == step),
                     key=lambda r: r["eval_accuracy"], reverse=True)[:5]
        print(f"s{step} top: " + ", ".join(
            f"{r['role']}@L{r['layer']}={r['eval_accuracy']:.3f}" for r in top
        ), flush=True)
    return rows


def plot(rows, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    for step, axis in enumerate(axes.flat, 1):
        matrix = np.asarray([
            [next(r["eval_accuracy"] for r in rows
                  if r["state_step"] == step and r["role"] == role and r["layer"] == layer)
             for layer in range(13)]
            for role in ROLE_NAMES
        ])
        image = axis.imshow(matrix, aspect="auto", vmin=0.125, vmax=1.0, cmap="viridis")
        axis.set_title(f"Ground-truth state s{step}")
        axis.set_xticks(range(13)); axis.set_yticks(range(len(ROLE_NAMES)), ROLE_NAMES)
        axis.set_xlabel("Representation layer")
    for axis in axes[:, 0]: axis.set_ylabel("Semantic token role")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Held-out probe accuracy", shrink=0.8)
    fig.savefig(output_dir / "position_state_map.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config_from_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    model, checkpoint_step = load_model(config, args.checkpoint, encoder_config, tokenizer, device)
    train_dataset, eval_dataset = load_dataset_split(config.data_path), load_dataset_split(config.eval_data_path)
    train_selected, train_ids = select_compose_d4(train_dataset, args.train_samples, args.seed)
    eval_selected, eval_ids = select_compose_d4(eval_dataset, args.eval_samples, args.seed + 1)
    train_features = extract(train_selected, model, encoder, tokenizer, config, device,
                             args.batch_size, args.seed + 2)
    eval_features = extract(eval_selected, model, encoder, tokenizer, config, device,
                            args.batch_size, args.seed + 3)
    train_states = np.stack(train_selected["states"]).astype(np.int64)
    eval_states = np.stack(eval_selected["states"]).astype(np.int64)
    alphas = args.alpha or [0.1, 1.0, 10.0, 100.0]
    metrics = fit_probes(train_features, train_states, eval_features, eval_states, alphas, args.seed)
    non_oracle_top = {}
    for step in range(1, 5):
        choices = [r for r in metrics if r["state_step"] == step and not r["oracle_input_anchor"]]
        non_oracle_top[f"s{step}"] = sorted(
            choices, key=lambda r: r["eval_accuracy"], reverse=True,
        )[:10]
    summary = {
        "checkpoint": args.checkpoint, "checkpoint_step": checkpoint_step,
        "device": str(device), "train_samples": args.train_samples,
        "eval_samples": args.eval_samples, "roles": ROLE_NAMES,
        "role_warning": "oracle_cell_k uses ground-truth path and is an input anchor, not latent-computation evidence",
        "alphas": alphas, "train_source_ids": train_ids, "eval_source_ids": eval_ids,
        "top_non_oracle": non_oracle_top, "metrics": metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "position_state_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    plot(metrics, output_dir)
    print(json.dumps(non_oracle_top, indent=2), flush=True)
    print(f"Saved position-state map to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
