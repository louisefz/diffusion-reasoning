#!/usr/bin/env python
"""Factorial Q/K/V patching for block-11 attention head 3."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "official-elf" / "src"))

from configs.config import load_config_from_yaml
from modules.t5_encoder import get_encoder
from official_counterfactual_patching import build_pairs, decode, prepare
from official_diagnostics import load_model
from official_layerwise_patching import semantic_token_map
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split


MODES = (
    "q_answer", "k_all", "v_all", "kv_all",
    "q_answer_k_all", "q_answer_v_all", "q_answer_kv_all",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pairs-per-step", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--block", type=int, default=11)
    parser.add_argument("--head", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260919)
    return parser.parse_args()


def capture_qkv(model, model_input, t, sc, block, use_bf16):
    captured = {}
    module = model.blocks[block - 1].attn.qkv

    def hook(_module, _inputs, output):
        captured["qkv"] = output.detach().clone()

    handle = module.register_forward_hook(hook)
    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            endpoint, _ = model(model_input, t, deterministic=True,
                                self_cond_cfg_scale=sc, decoder_step_active=None)
    finally:
        handle.remove()
    return endpoint, captured["qkv"]


def patched_qkv_forward(model, model_input, t, sc, source_positions, donor_positions,
                        block, head, donor_qkv, mode, use_bf16):
    attention = model.blocks[block - 1].attn
    prefix = model.num_model_mode_tokens + model.num_time_tokens + model.num_self_cond_cfg_tokens
    rows = torch.arange(model_input.shape[0], device=model_input.device)
    components = {
        "q_answer": {"q"},
        "k_all": {"k"},
        "v_all": {"v"},
        "kv_all": {"k", "v"},
        "q_answer_k_all": {"q", "k"},
        "q_answer_v_all": {"q", "v"},
        "q_answer_kv_all": {"q", "k", "v"},
    }[mode]
    patch_q, patch_k, patch_v = (
        "q" in components, "k" in components, "v" in components,
    )

    def hook(_module, _inputs, output):
        bsz, length, _ = output.shape
        patched = output.clone().reshape(
            bsz, length, 3, attention.num_heads,
            attention.dim // attention.num_heads,
        )
        donor = donor_qkv.reshape_as(patched)
        if patch_q:
            patched[rows, source_positions + prefix, 0, head] = donor[
                rows, donor_positions + prefix, 0, head
            ].to(patched.dtype)
        if patch_k:
            patched[:, :, 1, head] = donor[:, :, 1, head].to(patched.dtype)
        if patch_v:
            patched[:, :, 2, head] = donor[:, :, 2, head].to(patched.dtype)
        return patched.reshape_as(output)

    handle = attention.qkv.register_forward_hook(hook)
    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            endpoint, _ = model(model_input, t, deterministic=True,
                                self_cond_cfg_scale=sc, decoder_step_active=None)
    finally:
        handle.remove()
    return endpoint


def aggregate(rows):
    result = []
    for subset in ("all", "same_answer_position"):
        chosen = rows if subset == "all" else [r for r in rows if r["same_answer_position"]]
        buckets = defaultdict(list)
        for row in chosen:
            buckets[(row["intervention_step"], row["mode"])].append(row)
        for (step, mode), items in sorted(buckets.items()):
            result.append({
                "subset": subset, "intervention_step": step, "mode": mode,
                "samples": len(items),
                "counterfactual_answer_rate": float(np.mean([x["follows_counterfactual"] for x in items])),
                "original_answer_rate": float(np.mean([x["follows_original"] for x in items])),
                "other_answer_rate": float(np.mean([x["follows_other"] for x in items])),
                "original_baseline_correct": float(np.mean([x["original_baseline_correct"] for x in items])),
                "counterfactual_baseline_correct": float(np.mean([x["counterfactual_baseline_correct"] for x in items])),
            })
    return result


def plot(metrics, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in metrics if r["subset"] == "same_answer_position"]
    matrix = np.asarray([
        [next(r["counterfactual_answer_rate"] for r in rows
              if r["intervention_step"] == step and r["mode"] == mode)
         for mode in MODES]
        for step in range(1, 5)
    ])
    fig, axis = plt.subplots(figsize=(10, 4.5))
    image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks(range(len(MODES)), MODES, rotation=30, ha="right")
    axis.set_yticks(range(4), ["edit s1", "edit s2", "edit s3", "edit s4"])
    axis.set_title("Block-11 head-3 Q/K/V factorial patching (aligned answer positions)")
    fig.colorbar(image, ax=axis, label="Follows counterfactual answer")
    fig.tight_layout(); fig.savefig(output_dir / "head3_qkv_factorial.png", dpi=180); plt.close(fig)


@torch.no_grad()
def run(args):
    if args.pairs_per_step % args.batch_size:
        raise ValueError("--batch-size must divide --pairs-per-step")
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config_from_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    token_map = semantic_token_map(tokenizer)
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    model, checkpoint_step = load_model(config, args.checkpoint, encoder_config, tokenizer, device)
    validation = load_dataset_split(config.eval_data_path)
    originals, counterfactuals, counts = build_pairs(validation, tokenizer, args.pairs_per_step, args.seed)
    loader_args = dict(
        batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False,
        max_seq_length=config.max_length,
        pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
        max_input_seq_length=config.max_input_length, distributed=False,
    )
    original_loader = get_dataloader(originals, **loader_args)
    counterfactual_loader = get_dataloader(counterfactuals, **loader_args)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    dtype = next(model.parameters()).dtype
    use_bf16 = bool(config.use_bf16) and device.type == "cuda"
    rows = []; cursor = 0
    for source_batch, cf_batch in zip(original_loader, counterfactual_loader):
        bsz = len(source_batch["target"])
        noise = torch.randn((bsz, config.max_length, model.text_encoder_dim), generator=generator, dtype=dtype)
        source = prepare(source_batch, model, encoder, tokenizer, config, device, noise)
        cf = prepare(cf_batch, model, encoder, tokenizer, config, device, noise)
        source_endpoint, _ = capture_qkv(
            model, source["model_input"], source["t"], source["sc"], args.block, use_bf16,
        )
        cf_endpoint, donor_qkv = capture_qkv(
            model, cf["model_input"], cf["t"], cf["sc"], args.block, use_bf16,
        )
        source_stats, cf_stats = decode(source_endpoint, source, model, config), decode(cf_endpoint, cf, model, config)
        source_tokens = torch.tensor([token_map[int(x)] for x in source_batch["target"]], device=device)
        cf_tokens = torch.tensor([token_map[int(x)] for x in cf_batch["target"]], device=device)
        steps = list(map(int, originals[cursor:cursor + bsz]["intervention_step"]))
        aligned = (source["positions"] == cf["positions"]).cpu().tolist()
        for mode in MODES:
            endpoint = patched_qkv_forward(
                model, source["model_input"], source["t"], source["sc"],
                source["positions"], cf["positions"], args.block, args.head,
                donor_qkv, mode, use_bf16,
            )
            stats = decode(endpoint, source, model, config)
            for local in range(bsz):
                prediction = stats["prediction"][local]
                follows_source = bool(prediction == source_tokens[local])
                follows_cf = bool(prediction == cf_tokens[local])
                rows.append({
                    "pair_index": cursor + local, "intervention_step": steps[local],
                    "mode": mode, "same_answer_position": bool(aligned[local]),
                    "follows_original": follows_source, "follows_counterfactual": follows_cf,
                    "follows_other": not follows_source and not follows_cf,
                    "original_baseline_correct": bool(source_stats["correct"][local]),
                    "counterfactual_baseline_correct": bool(cf_stats["correct"][local]),
                })
        cursor += bsz; print(f"Patched {cursor}/{len(originals)} pairs", flush=True)
    metrics = aggregate(rows)
    summary = {
        "checkpoint": args.checkpoint, "checkpoint_step": checkpoint_step,
        "block": args.block, "head": args.head, "counts": counts,
        "modes": MODES,
        "note": "Q is patched at answer query; K/V are patched for the whole fixed-length sequence",
        "aggregate": metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "qkv_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    plot(metrics, output_dir)
    print(f"Saved QKV factorial artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    run(parse_args())
