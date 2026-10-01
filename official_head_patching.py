#!/usr/bin/env python
"""Head-level causal patching for block-11 counterfactual answer formation."""

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
from official_counterfactual_patching import build_pairs_for_depth, decode, prepare
from official_diagnostics import load_model
from official_intermediate_state_probe import select_compose_d4  # noqa: F401
from official_layerwise_patching import _capture_first_forward, semantic_token_map
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pairs-per-step", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--block", type=int, default=11)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260919)
    return parser.parse_args()


def capture_preprojection(model, model_input, t, sc, positions, block, use_bf16):
    attention = model.blocks[block - 1].attn
    rows = torch.arange(model_input.shape[0], device=model_input.device)
    prefix = model.num_model_mode_tokens + model.num_time_tokens + model.num_self_cond_cfg_tokens
    captured = {}

    def hook(_module, inputs):
        concatenated = inputs[0]
        vector = concatenated[rows, positions + prefix]
        captured["heads"] = vector.reshape(
            vector.shape[0], attention.num_heads, attention.dim // attention.num_heads,
        ).detach().clone()

    handle = attention.proj.register_forward_pre_hook(hook)
    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            endpoint, _ = model(model_input, t, deterministic=True,
                                self_cond_cfg_scale=sc, decoder_step_active=None)
    finally:
        handle.remove()
    return endpoint, captured["heads"]


def patched_head_forward(model, model_input, t, sc, positions, block,
                         source_heads, donor_heads, mode, head, use_bf16):
    attention = model.blocks[block - 1].attn
    rows = torch.arange(model_input.shape[0], device=model_input.device)
    prefix = model.num_model_mode_tokens + model.num_time_tokens + model.num_self_cond_cfg_tokens
    head_dim = attention.dim // attention.num_heads

    def hook(_module, inputs):
        output = inputs[0].clone()
        vectors = output[rows, positions + prefix].reshape(
            output.shape[0], attention.num_heads, head_dim,
        )
        if mode == "individual":
            vectors[:, head] = donor_heads[:, head].to(vectors.dtype)
        elif mode == "all_except":
            vectors[:] = donor_heads.to(vectors.dtype)
            vectors[:, head] = source_heads[:, head].to(vectors.dtype)
        elif mode == "all":
            vectors[:] = donor_heads.to(vectors.dtype)
        else:
            raise ValueError(mode)
        output[rows, positions + prefix] = vectors.reshape(output.shape[0], attention.dim)
        return (output,)

    handle = attention.proj.register_forward_pre_hook(hook)
    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            endpoint, _ = model(model_input, t, deterministic=True,
                                self_cond_cfg_scale=sc, decoder_step_active=None)
    finally:
        handle.remove()
    return endpoint


def aggregate(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["intervention_step"], row["mode"], row["head"])].append(row)
    result = []
    for (step, mode, head), items in sorted(buckets.items()):
        result.append({
            "intervention_step": step, "mode": mode, "head": head,
            "samples": len(items),
            "counterfactual_answer_rate": float(np.mean([x["follows_counterfactual"] for x in items])),
            "original_answer_rate": float(np.mean([x["follows_original"] for x in items])),
            "other_answer_rate": float(np.mean([x["follows_other"] for x in items])),
            "original_baseline_correct": float(np.mean([x["original_baseline_correct"] for x in items])),
            "counterfactual_baseline_correct": float(np.mean([x["counterfactual_baseline_correct"] for x in items])),
        })
    return result


def plot(metrics, output_dir, num_heads, depth, block):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for axis, mode, title in (
        (axes[0], "individual", "Single counterfactual head (sufficiency)"),
        (axes[1], "all_except", "All counterfactual except one source head"),
    ):
        matrix = np.asarray([
            [next(r["counterfactual_answer_rate"] for r in metrics
                  if r["intervention_step"] == step and r["mode"] == mode and r["head"] == head)
             for head in range(num_heads)]
            for step in range(1, depth + 1)
        ])
        image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        axis.set_title(title); axis.set_xlabel(f"Block-{block} attention head")
        axis.set_ylabel("Counterfactual edit step"); axis.set_xticks(range(num_heads))
        axis.set_yticks(range(depth), [f"s{step}" for step in range(1, depth + 1)])
        fig.colorbar(image, ax=axis, label="Follows counterfactual")
    fig.tight_layout(); fig.savefig(output_dir / "head_causal_patching.png", dpi=180); plt.close(fig)


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
    originals, counterfactuals, counts = build_pairs_for_depth(
        validation, tokenizer, args.pairs_per_step, args.depth, args.seed)
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
    num_heads = model.blocks[args.block - 1].attn.num_heads
    rows = []; cursor = 0
    for source_batch, cf_batch in zip(original_loader, counterfactual_loader):
        bsz = len(source_batch["target"])
        noise = torch.randn((bsz, config.max_length, model.text_encoder_dim), generator=generator, dtype=dtype)
        source = prepare(source_batch, model, encoder, tokenizer, config, device, noise)
        cf = prepare(cf_batch, model, encoder, tokenizer, config, device, noise)
        source_endpoint, source_heads = capture_preprojection(
            model, source["model_input"], source["t"], source["sc"], source["positions"],
            args.block, use_bf16,
        )
        cf_endpoint, donor_heads = capture_preprojection(
            model, cf["model_input"], cf["t"], cf["sc"], cf["positions"],
            args.block, use_bf16,
        )
        source_stats, cf_stats = decode(source_endpoint, source, model, config), decode(cf_endpoint, cf, model, config)
        source_tokens = torch.tensor([token_map[int(x)] for x in source_batch["target"]], device=device)
        cf_tokens = torch.tensor([token_map[int(x)] for x in cf_batch["target"]], device=device)
        steps = list(map(int, originals[cursor:cursor + bsz]["intervention_step"]))
        interventions = []
        for head in range(num_heads):
            for mode in ("individual", "all_except"):
                endpoint = patched_head_forward(
                    model, source["model_input"], source["t"], source["sc"], source["positions"],
                    args.block, source_heads, donor_heads, mode, head, use_bf16,
                )
                interventions.append((mode, head, decode(endpoint, source, model, config)))
        endpoint = patched_head_forward(
            model, source["model_input"], source["t"], source["sc"], source["positions"],
            args.block, source_heads, donor_heads, "all", 0, use_bf16,
        )
        interventions.append(("all", -1, decode(endpoint, source, model, config)))
        for mode, head, stats in interventions:
            for local in range(bsz):
                prediction = stats["prediction"][local]
                follows_source = bool(prediction == source_tokens[local])
                follows_cf = bool(prediction == cf_tokens[local])
                rows.append({
                    "pair_index": cursor + local, "intervention_step": steps[local],
                    "mode": mode, "head": head,
                    "follows_original": follows_source, "follows_counterfactual": follows_cf,
                    "follows_other": not follows_source and not follows_cf,
                    "original_baseline_correct": bool(source_stats["correct"][local]),
                    "counterfactual_baseline_correct": bool(cf_stats["correct"][local]),
                })
        cursor += bsz; print(f"Patched {cursor}/{len(originals)} pairs", flush=True)
    metrics = aggregate(rows)
    summary = {
        "checkpoint": args.checkpoint, "checkpoint_step": checkpoint_step,
        "block": args.block, "depth": args.depth,
        "num_heads": num_heads, "counts": counts,
        "individual_definition": "replace one source pre-projection head output with paired counterfactual head",
        "all_except_definition": "replace all heads with counterfactual except the named source head",
        "aggregate": metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "head_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    plot(metrics, output_dir, num_heads, args.depth, args.block)
    print(f"Saved head patching artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    run(parse_args())
