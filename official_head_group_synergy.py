#!/usr/bin/env python
"""Measure cooperative answer control by groups of attention heads.

The shallow composition tasks carry causal answer information in block-10
attention, but no single head is sufficient.  This experiment patches selected
sets of pre-projection head outputs at the answer position and quantifies both
discrete donor-answer control and continuous donor-vs-source logit response.
"""

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
from official_head_patching import capture_preprojection
from official_layerwise_patching import semantic_token_map
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split
from utils.generation_utils import _dlm_decode_logits


GROUPS = {
    "none": (),
    "h1": (1,), "h2": (2,), "h3": (3,),
    "h1_h2": (1, 2), "h1_h3": (1, 3), "h2_h3": (2, 3),
    "h1_h2_h3": (1, 2, 3),
    "control_h4": (4,), "control_h4_h5": (4, 5),
    "control_h4_h5_h6": (4, 5, 6),
    "all": tuple(range(12)),
    "all_except_h3": tuple(head for head in range(12) if head != 3),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--depth", type=int, required=True, choices=(1, 2))
    parser.add_argument("--block", type=int, default=10)
    parser.add_argument("--pairs-per-step", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260929)
    return parser.parse_args()


def patched_group_forward(model, model_input, t, sc, positions, block,
                          donor_heads, heads, use_bf16):
    attention = model.blocks[block - 1].attn
    rows = torch.arange(model_input.shape[0], device=model_input.device)
    prefix = model.num_model_mode_tokens + model.num_time_tokens + model.num_self_cond_cfg_tokens
    head_dim = attention.dim // attention.num_heads

    def hook(_module, inputs):
        output = inputs[0].clone()
        vectors = output[rows, positions + prefix].reshape(
            output.shape[0], attention.num_heads, head_dim)
        if heads:
            selected = list(heads)
            vectors[:, selected] = donor_heads[:, selected].to(vectors.dtype)
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


def logit_contrast(endpoint, prepared, model, config, source_tokens, donor_tokens):
    logits = _dlm_decode_logits(endpoint, model, 1.0, config, 1.0)
    rows = torch.arange(endpoint.shape[0], device=endpoint.device)
    answer_logits = logits[rows, prepared["positions"]]
    source = answer_logits.gather(1, source_tokens[:, None]).squeeze(1)
    donor = answer_logits.gather(1, donor_tokens[:, None]).squeeze(1)
    return (donor - source).float()


def aggregate(rows):
    result = []
    for subset in ("all", "eligible"):
        chosen = rows if subset == "all" else [row for row in rows if row["eligible"]]
        buckets = defaultdict(list)
        for row in chosen:
            buckets[(row["intervention_step"], row["group"])].append(row)
        for (step, group), items in sorted(buckets.items()):
            result.append({
                "subset": subset, "intervention_step": step, "group": group,
                "heads": list(GROUPS[group]), "samples": len(items),
                "counterfactual_answer_rate": float(np.mean(
                    [item["follows_counterfactual"] for item in items])),
                "original_answer_rate": float(np.mean(
                    [item["follows_original"] for item in items])),
                "logit_contrast": float(np.mean(
                    [item["logit_contrast"] for item in items])),
                "logit_response": float(np.mean(
                    [item["logit_response"] for item in items])),
            })
    return result


def synergy_metrics(metrics, depth):
    lookup = {(row["intervention_step"], row["group"]): row
              for row in metrics if row["subset"] == "eligible"}
    result = []
    for step in range(1, depth + 1):
        def value(group, field):
            return lookup[(step, group)][field]
        for field in ("counterfactual_answer_rate", "logit_response"):
            baseline = value("none", field)
            for pair, left, right in (
                ("h1_h2", "h1", "h2"),
                ("h1_h3", "h1", "h3"),
                ("h2_h3", "h2", "h3"),
            ):
                result.append({
                    "intervention_step": step, "field": field,
                    "interaction": pair,
                    "synergy": value(pair, field) - value(left, field)
                               - value(right, field) + baseline,
                })
            triple = (value("h1_h2_h3", field)
                      - value("h1_h2", field) - value("h1_h3", field)
                      - value("h2_h3", field)
                      + value("h1", field) + value("h2", field)
                      + value("h3", field) - baseline)
            result.append({
                "intervention_step": step, "field": field,
                "interaction": "h1_h2_h3_mobius", "synergy": triple,
            })
    return result


def plot(metrics, output_dir, depth, block):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = list(GROUPS)
    eligible = [row for row in metrics if row["subset"] == "eligible"]
    fig, axes = plt.subplots(depth, 2, figsize=(15, 4.3 * depth), squeeze=False)
    for step in range(1, depth + 1):
        chosen = {row["group"]: row for row in eligible
                  if row["intervention_step"] == step}
        for axis, field, title in (
            (axes[step - 1, 0], "counterfactual_answer_rate", "Donor-answer control"),
            (axes[step - 1, 1], "logit_response", "Donor-vs-source logit response"),
        ):
            values = [chosen[group][field] for group in order]
            axis.bar(range(len(order)), values)
            axis.axhline(0, color="black", linewidth=.8)
            axis.set_xticks(range(len(order)), order, rotation=35, ha="right")
            axis.set_title(f"d{depth}, edit s{step}: {title}")
            axis.grid(axis="y", alpha=.25)
            if field == "counterfactual_answer_rate": axis.set_ylim(-.03, 1.03)
    fig.suptitle(f"Block-{block} head-group causal synergy", y=.995)
    fig.tight_layout()
    fig.savefig(output_dir / "head_group_synergy.png", dpi=220)
    plt.close(fig)


@torch.no_grad()
def run(args):
    if args.pairs_per_step % args.batch_size:
        raise ValueError("--batch-size must divide --pairs-per-step")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config_from_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    token_map = semantic_token_map(tokenizer)
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    model, checkpoint_step = load_model(config, args.checkpoint, encoder_config, tokenizer, device)
    if model.blocks[args.block - 1].attn.num_heads != 12:
        raise ValueError("GROUPS currently assumes the 12-head ELF checkpoint")
    validation = load_dataset_split(config.eval_data_path)
    originals, counterfactuals, counts = build_pairs_for_depth(
        validation, tokenizer, args.pairs_per_step, args.depth, args.seed)
    loader_args = dict(
        batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False,
        max_seq_length=config.max_length,
        pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
        max_input_seq_length=config.max_input_length, distributed=False)
    source_loader = get_dataloader(originals, **loader_args)
    donor_loader = get_dataloader(counterfactuals, **loader_args)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    dtype = next(model.parameters()).dtype
    use_bf16 = bool(config.use_bf16) and device.type == "cuda"
    rows = []
    cursor = 0

    for source_batch, donor_batch in zip(source_loader, donor_loader):
        bsz = len(source_batch["target"])
        noise = torch.randn((bsz, config.max_length, model.text_encoder_dim),
                            generator=generator, dtype=dtype)
        source = prepare(source_batch, model, encoder, tokenizer, config, device, noise)
        donor = prepare(donor_batch, model, encoder, tokenizer, config, device, noise)
        source_endpoint, _ = capture_preprojection(
            model, source["model_input"], source["t"], source["sc"],
            source["positions"], args.block, use_bf16)
        donor_endpoint, donor_heads = capture_preprojection(
            model, donor["model_input"], donor["t"], donor["sc"],
            donor["positions"], args.block, use_bf16)
        source_stats = decode(source_endpoint, source, model, config)
        donor_stats = decode(donor_endpoint, donor, model, config)
        source_tokens = torch.tensor(
            [token_map[int(x)] for x in source_batch["target"]], device=device)
        donor_tokens = torch.tensor(
            [token_map[int(x)] for x in donor_batch["target"]], device=device)
        source_contrast = logit_contrast(
            source_endpoint, source, model, config, source_tokens, donor_tokens)
        steps = list(map(int, originals[cursor:cursor + bsz]["intervention_step"]))
        eligible = source_stats["correct"] & donor_stats["correct"]

        for group, heads in GROUPS.items():
            endpoint = patched_group_forward(
                model, source["model_input"], source["t"], source["sc"],
                source["positions"], args.block, donor_heads, heads, use_bf16)
            stats = decode(endpoint, source, model, config)
            contrast = logit_contrast(
                endpoint, source, model, config, source_tokens, donor_tokens)
            for local in range(bsz):
                prediction = stats["prediction"][local]
                rows.append({
                    "pair_index": cursor + local,
                    "intervention_step": steps[local], "group": group,
                    "eligible": bool(eligible[local]),
                    "follows_original": bool(prediction == source_tokens[local]),
                    "follows_counterfactual": bool(prediction == donor_tokens[local]),
                    "logit_contrast": float(contrast[local]),
                    "logit_response": float(contrast[local] - source_contrast[local]),
                })
        cursor += bsz
        print(f"Head-group synergy: processed {cursor}/{len(originals)}", flush=True)

    metrics = aggregate(rows)
    synergy = synergy_metrics(metrics, args.depth)
    summary = {
        "checkpoint": args.checkpoint, "checkpoint_step": checkpoint_step,
        "depth": args.depth, "block": args.block, "counts": counts,
        "groups": {name: list(heads) for name, heads in GROUPS.items()},
        "metrics": metrics, "synergy": synergy,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "head_group_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader(); writer.writerows(metrics)
    with (output_dir / "per_sample.jsonl").open("w") as handle:
        for row in rows: handle.write(json.dumps(row) + "\n")
    plot(metrics, output_dir, args.depth, args.block)
    print(f"Saved head-group synergy to {output_dir}", flush=True)


if __name__ == "__main__":
    run(parse_args())
