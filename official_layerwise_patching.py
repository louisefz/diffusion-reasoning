#!/usr/bin/env python
"""Causally patch answer-slot activations across ELF-B denoiser layers."""

import argparse
from collections import defaultdict
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent
OFFICIAL_SRC = ROOT / "official-elf" / "src"
sys.path.insert(0, str(OFFICIAL_SRC))

from configs.config import load_config_from_yaml
from modules.t5_encoder import get_encoder
from official_diagnostics import load_model
from official_flow_trajectory import answer_decoder_stats
from official_layerwise_probe import HiddenRecorder, select_balanced
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split
from utils.encoder_utils import encode_text
from utils.generation_utils import _dlm_decode_logits
from utils.sampling_utils import restore_cond


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-group", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260919)
    return parser.parse_args()


def donor_indices(labels, same_answer, seed):
    """Choose a non-self donor for every row, within or across answer classes."""
    labels = [int(value) for value in labels]
    rng = random.Random(seed)
    donors = []
    for row, label in enumerate(labels):
        candidates = [
            index for index, other in enumerate(labels)
            if index != row and ((other == label) == same_answer)
        ]
        # A small smoke batch can contain a singleton answer class.  Its
        # same-answer control becomes an explicit no-op rather than failing.
        if same_answer and not candidates:
            candidates = [row]
        if not candidates:
            raise ValueError(
                f"No {'same' if same_answer else 'different'}-answer donor "
                f"for row {row}; increase batch size"
            )
        donors.append(rng.choice(candidates))
    return torch.tensor(donors, dtype=torch.long)


def semantic_token_map(tokenizer):
    return {
        value: tokenizer.encode(str(value), add_special_tokens=False)[-1]
        for value in range(8)
    }


def _capture_first_forward(model, model_input, t_batch, sc_batch, answer_positions,
                           use_bf16):
    recorder = HiddenRecorder(model, answer_positions)
    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            endpoint, _ = model(
                model_input, t_batch, deterministic=True,
                self_cond_cfg_scale=sc_batch, decoder_step_active=None,
            )
    finally:
        recorder.close()
    return endpoint, torch.stack(recorder.hidden, dim=1)


def _patched_forward(model, model_input, t_batch, sc_batch, answer_positions,
                     donor_vectors, layer, alpha, use_bf16):
    rows = torch.arange(model_input.shape[0], device=model_input.device)
    prefix = (
        model.num_model_mode_tokens + model.num_time_tokens
        + model.num_self_cond_cfg_tokens
    )

    def patch_hook(_module, _inputs, output):
        patched = output.clone()
        positions = answer_positions if layer == 0 else answer_positions + prefix
        source = patched[rows, positions]
        patched[rows, positions] = (
            (1.0 - alpha) * source + alpha * donor_vectors.to(source.dtype)
        )
        return patched

    module = model.text_proj if layer == 0 else model.blocks[layer - 1]
    handle = module.register_forward_hook(patch_hook)
    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            endpoint, _ = model(
                model_input, t_batch, deterministic=True,
                self_cond_cfg_scale=sc_batch, decoder_step_active=None,
            )
    finally:
        handle.remove()
    return endpoint


def aggregate(per_sample):
    buckets = defaultdict(list)
    for row in per_sample:
        buckets[(row["group"], row["donor_type"], row["layer"])].append(row)
    result = []
    for (group, donor_type, layer), rows in sorted(buckets.items()):
        result.append({
            "group": group,
            "donor_type": donor_type,
            "layer": layer,
            "samples": len(rows),
            "source_answer_rate": float(np.mean([
                row["follows_source"] for row in rows
            ])),
            "donor_answer_rate": float(np.mean([
                row["follows_donor"] for row in rows
            ])),
            "other_answer_rate": float(np.mean([
                row["follows_other"] for row in rows
            ])),
            "exact_correct_rate": float(np.mean([
                row["exact_correct"] for row in rows
            ])),
            "baseline_exact_correct_rate": float(np.mean([
                row["baseline_exact_correct"] for row in rows
            ])),
        })
    return result


def _plot(rows, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = sorted({row["group"] for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for axis, group in zip(axes.flat, groups):
        different = [row for row in rows
                     if row["group"] == group and row["donor_type"] == "different"]
        same = [row for row in rows
                if row["group"] == group and row["donor_type"] == "same"]
        axis.plot(
            [row["layer"] for row in different],
            [row["donor_answer_rate"] for row in different],
            marker="o", label="different-answer donor: follows donor",
        )
        axis.plot(
            [row["layer"] for row in different],
            [row["source_answer_rate"] for row in different],
            marker="o", label="different-answer donor: follows source",
        )
        axis.plot(
            [row["layer"] for row in same],
            [row["exact_correct_rate"] for row in same],
            marker="o", linestyle="--", label="same-answer control: correct",
        )
        axis.set_title(group)
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("Patched representation (0=input projection, 1–12=blocks)")
        axis.set_xticks(range(13))
    for axis in axes[:, 0]:
        axis.set_ylabel("Fraction")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "layerwise_causal_patching.png", dpi=180)
    plt.close(fig)


@torch.no_grad()
def run(args):
    if args.samples_per_group % args.batch_size:
        raise ValueError("--batch-size must divide --samples-per-group")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config_from_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name or config.encoder_model_name
    )
    token_map = semantic_token_map(tokenizer)
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    model, checkpoint_step = load_model(
        config, args.checkpoint, encoder_config, tokenizer, device,
    )
    dataset = load_dataset_split(config.eval_data_path)
    selected, expected_groups, source_ids = select_balanced(
        dataset, args.samples_per_group, args.seed,
    )
    loader = get_dataloader(
        selected, batch_size=args.batch_size, shuffle=False, num_workers=0,
        drop_last=False, max_seq_length=config.max_length,
        pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
        max_input_seq_length=config.max_input_length, distributed=False,
    )
    dtype = next(model.parameters()).dtype
    noise_generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    use_bf16 = bool(config.use_bf16) and device.type == "cuda"
    per_sample = []
    cursor = 0

    for batch_index, batch in enumerate(loader):
        bsz = len(batch["target"])
        batch_groups = expected_groups[cursor:cursor + bsz]
        if len(set(batch_groups)) != 1:
            raise ValueError(f"Batch crosses groups: {set(batch_groups)}")
        group = batch_groups[0]
        labels = [int(target) for target in batch["target"]]
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
        baseline_endpoint, hidden = _capture_first_forward(
            model, model_input, t_batch, sc_batch, answer_positions, use_bf16,
        )
        baseline_logits = _dlm_decode_logits(
            baseline_endpoint, model, 1.0, config, 1.0,
        )
        baseline_stats = answer_decoder_stats(
            baseline_logits, answer_starts, answer_lengths, input_ids,
        )

        for donor_type, same_answer in (("different", False), ("same", True)):
            donors = donor_indices(
                labels, same_answer=same_answer,
                seed=args.seed + batch_index * 17 + int(same_answer),
            ).to(device)
            donor_labels = [labels[int(index)] for index in donors]
            source_tokens = torch.tensor(
                [token_map[label] for label in labels], device=device,
            )
            donor_tokens = torch.tensor(
                [token_map[label] for label in donor_labels], device=device,
            )
            for layer in range(model.depth + 1):
                donor_vectors = hidden[donors, layer]
                endpoint = _patched_forward(
                    model, model_input, t_batch, sc_batch, answer_positions,
                    donor_vectors, layer, args.alpha, use_bf16,
                )
                logits = _dlm_decode_logits(endpoint, model, 1.0, config, 1.0)
                stats = answer_decoder_stats(
                    logits, answer_starts, answer_lengths, input_ids,
                )
                prediction = stats["prediction"]
                for local in range(bsz):
                    follows_source = bool(prediction[local] == source_tokens[local])
                    follows_donor = bool(prediction[local] == donor_tokens[local])
                    per_sample.append({
                        "source_id": source_ids[cursor + local],
                        "group": group,
                        "source_answer": labels[local],
                        "donor_answer": donor_labels[local],
                        "donor_type": donor_type,
                        "layer": layer,
                        "alpha": args.alpha,
                        "prediction_token_id": int(prediction[local]),
                        "follows_source": follows_source,
                        "follows_donor": follows_donor,
                        "follows_other": not follows_source and not follows_donor,
                        "exact_correct": bool(stats["correct"][local]),
                        "baseline_exact_correct": bool(
                            baseline_stats["correct"][local]
                        ),
                    })
        cursor += bsz
        print(f"Patched {cursor}/{len(selected)} samples", flush=True)

    aggregate_rows = aggregate(per_sample)
    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": checkpoint_step,
        "device": str(device),
        "samples_per_group": args.samples_per_group,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "alpha": args.alpha,
        "layers": model.depth,
        "scope": "semantic answer-token hidden state in first conditional denoiser forward",
        "aggregate": aggregate_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    with (output_dir / "per_sample.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_sample:
            handle.write(json.dumps(row) + "\n")
    with (output_dir / "patching_metrics.csv").open(
        "w", newline="", encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    _plot(aggregate_rows, output_dir)
    print(f"Saved layer patching artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    run(parse_args())
