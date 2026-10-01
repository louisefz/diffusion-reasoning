#!/usr/bin/env python
"""Trace when counterfactual routing is written into final-table positions."""

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
from official_head_position_patching import semantic_positions
from official_layerwise_patching import semantic_token_map
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split


MODES = ("final_function_table", "changed_function_table", "all_table")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pairs-per-step", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-layer", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260919)
    return parser.parse_args()


def semantic_mappings(source_row, cf_row, tokenizer):
    source, donor = semantic_positions(source_row, tokenizer), semantic_positions(cf_row, tokenizer)
    step = int(cf_row["intervention_step"])
    program = list(map(int, source_row["program"]))
    edited_function, final_function = program[step - 1], program[3]

    def table(function):
        return [(source["table"][(function, value)], donor["table"][(function, value)])
                for value in range(8)]

    return {
        "final_function_table": table(final_function),
        "changed_function_table": table(edited_function),
        "all_table": [(source["table"][key], donor["table"][key])
                      for key in sorted(source["table"])],
    }


def capture_hidden(model, model_input, t, sc, max_layer, use_bf16):
    hidden, handles = [], []

    def hook(_module, _inputs, output):
        hidden.append(output.detach().clone())

    handles.append(model.text_proj.register_forward_hook(hook))
    for block in model.blocks[:max_layer]:
        handles.append(block.register_forward_hook(hook))
    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            endpoint, _ = model(model_input, t, deterministic=True,
                                self_cond_cfg_scale=sc, decoder_step_active=None)
    finally:
        for handle in handles: handle.remove()
    if len(hidden) != max_layer + 1:
        raise RuntimeError(f"Captured {len(hidden)} layers, expected {max_layer + 1}")
    return endpoint, hidden


def patched_layer_forward(model, model_input, t, sc, layer, donor_hidden,
                          mappings, use_bf16):
    prefix = model.num_model_mode_tokens + model.num_time_tokens + model.num_self_cond_cfg_tokens
    module = model.text_proj if layer == 0 else model.blocks[layer - 1]

    def hook(_module, _inputs, output):
        patched = output.clone()
        offset = 0 if layer == 0 else prefix
        for row, pairs in enumerate(mappings):
            for source_position, donor_position in pairs:
                patched[row, source_position + offset] = donor_hidden[
                    row, donor_position + offset
                ].to(patched.dtype)
        return patched

    handle = module.register_forward_hook(hook)
    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            endpoint, _ = model(model_input, t, deterministic=True,
                                self_cond_cfg_scale=sc, decoder_step_active=None)
    finally:
        handle.remove()
    return endpoint


def aggregate(rows):
    result = []
    subsets = {
        "all": rows,
        "edited_function_not_final": [r for r in rows if not r["edited_function_is_final"]],
    }
    for subset, selected in subsets.items():
        buckets = defaultdict(list)
        for row in selected:
            buckets[(row["intervention_step"], row["mode"], row["layer"])].append(row)
        for (step, mode, layer), items in sorted(buckets.items()):
            result.append({
                "subset": subset, "intervention_step": step, "mode": mode,
                "layer": layer, "samples": len(items),
                "counterfactual_answer_rate": float(np.mean([x["follows_counterfactual"] for x in items])),
                "original_answer_rate": float(np.mean([x["follows_original"] for x in items])),
                "other_answer_rate": float(np.mean([x["follows_other"] for x in items])),
                "original_baseline_correct": float(np.mean([x["original_baseline_correct"] for x in items])),
                "counterfactual_baseline_correct": float(np.mean([x["counterfactual_baseline_correct"] for x in items])),
            })
    return result


def plot(metrics, output_dir, max_layer):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in metrics if r["subset"] == "edited_function_not_final"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for axis, mode in zip(axes, MODES):
        for step in (1, 2, 3):
            chosen = [r for r in rows if r["mode"] == mode and r["intervention_step"] == step]
            axis.plot([r["layer"] for r in chosen],
                      [r["counterfactual_answer_rate"] for r in chosen],
                      marker="o", label=f"edit s{step}")
        axis.set_title(mode); axis.set_xlabel("Patched representation layer")
        axis.set_xticks(range(max_layer + 1)); axis.grid(alpha=.25); axis.set_ylim(-.03, 1.03)
    axes[0].set_ylabel("Follows counterfactual answer"); axes[0].legend()
    fig.tight_layout(); fig.savefig(output_dir / "routing_layer_patching.png", dpi=180); plt.close(fig)


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
    source_loader = get_dataloader(originals, **loader_args)
    donor_loader = get_dataloader(counterfactuals, **loader_args)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    dtype = next(model.parameters()).dtype
    use_bf16 = bool(config.use_bf16) and device.type == "cuda"
    rows, cursor = [], 0
    for source_batch, donor_batch in zip(source_loader, donor_loader):
        bsz = len(source_batch["target"])
        noise = torch.randn((bsz, config.max_length, model.text_encoder_dim), generator=generator, dtype=dtype)
        source = prepare(source_batch, model, encoder, tokenizer, config, device, noise)
        donor = prepare(donor_batch, model, encoder, tokenizer, config, device, noise)
        source_endpoint, _ = capture_hidden(
            model, source["model_input"], source["t"], source["sc"], args.max_layer, use_bf16,
        )
        donor_endpoint, donor_hidden = capture_hidden(
            model, donor["model_input"], donor["t"], donor["sc"], args.max_layer, use_bf16,
        )
        source_stats, donor_stats = decode(source_endpoint, source, model, config), decode(donor_endpoint, donor, model, config)
        source_tokens = torch.tensor([token_map[int(x)] for x in source_batch["target"]], device=device)
        donor_tokens = torch.tensor([token_map[int(x)] for x in donor_batch["target"]], device=device)
        source_rows = [originals[i] for i in range(cursor, cursor + bsz)]
        donor_rows = [counterfactuals[i] for i in range(cursor, cursor + bsz)]
        mappings = {mode: [] for mode in MODES}
        for local in range(bsz):
            current = semantic_mappings(source_rows[local], donor_rows[local], tokenizer)
            for mode in MODES: mappings[mode].append(current[mode])
        for mode in MODES:
            for layer in range(args.max_layer + 1):
                endpoint = patched_layer_forward(
                    model, source["model_input"], source["t"], source["sc"], layer,
                    donor_hidden[layer], mappings[mode], use_bf16,
                )
                stats = decode(endpoint, source, model, config)
                for local in range(bsz):
                    prediction = stats["prediction"][local]
                    follows_source = bool(prediction == source_tokens[local])
                    follows_donor = bool(prediction == donor_tokens[local])
                    step = int(donor_rows[local]["intervention_step"])
                    program = list(map(int, source_rows[local]["program"]))
                    rows.append({
                        "pair_index": cursor + local, "intervention_step": step,
                        "mode": mode, "layer": layer,
                        "edited_function_is_final": program[step - 1] == program[3],
                        "follows_original": follows_source,
                        "follows_counterfactual": follows_donor,
                        "follows_other": not follows_source and not follows_donor,
                        "original_baseline_correct": bool(source_stats["correct"][local]),
                        "counterfactual_baseline_correct": bool(donor_stats["correct"][local]),
                    })
        cursor += bsz; print(f"Patched {cursor}/{len(originals)} pairs", flush=True)
    metrics = aggregate(rows)
    summary = {
        "checkpoint": args.checkpoint, "checkpoint_step": checkpoint_step,
        "counts": counts, "max_layer": args.max_layer, "modes": MODES,
        "patch_definition": "replace semantic table-position residual states with paired counterfactual states",
        "aggregate": metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "routing_layer_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    plot(metrics, output_dir, args.max_layer)
    print(f"Saved routing-layer artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    run(parse_args())
