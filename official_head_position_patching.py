#!/usr/bin/env python
"""Patch block-11 head-3 value vectors by semantic source position."""

import argparse
import csv
import json
import re
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
    "changed_cell", "swap_partner", "both_changed", "all_table",
    "changed_function_table", "final_function_table",
    "step1_table", "step2_table", "step3_table", "step4_table",
    "program", "start", "answer", "all_semantic",
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
    parser.add_argument("--component", choices=("k", "v"), default="v")
    parser.add_argument("--seed", type=int, default=20260919)
    return parser.parse_args()


def final_overlapping_token(offsets, span):
    left, right = span
    indices = [i for i, (a, b) in enumerate(offsets)
               if b > a and a < right and b > left]
    if not indices:
        raise ValueError(f"No token overlaps {span}")
    return indices[-1]


def semantic_positions(row, tokenizer):
    prompt = str(row["input"])
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded["input_ids"]) != list(map(int, row["condition_input_ids"])):
        raise ValueError("Retokenization mismatch")
    offsets = encoded["offset_mapping"]
    table = {}
    for match in re.finditer(r"F([0-3]): ([0-7](?: [0-7]){7})", prompt):
        fn = int(match.group(1)); base = match.start(2)
        for input_value, value in enumerate(re.finditer(r"[0-7]", match.group(2))):
            table[(fn, input_value)] = final_overlapping_token(
                offsets, (base + value.start(), base + value.end()),
            )
    start_match = re.search(r"Start ([0-7])\.", prompt)
    program_match = re.search(r"Program (F[0-3](?: F[0-3])*)\.", prompt)
    if len(table) != 32 or start_match is None or program_match is None:
        raise ValueError("Could not parse semantic positions")
    base = program_match.start(1)
    program = [final_overlapping_token(offsets, (base + m.start(), base + m.end()))
               for m in re.finditer(r"F[0-3]", program_match.group(1))]
    return {
        "table": table,
        "program": program,
        "start": final_overlapping_token(offsets, start_match.span(1)),
    }


def position_mappings(source_row, cf_row, source_answer, cf_answer, tokenizer):
    source_pos, cf_pos = semantic_positions(source_row, tokenizer), semantic_positions(cf_row, tokenizer)
    step = int(cf_row["intervention_step"])
    fn = int(source_row["program"][step - 1])
    accessed = int(source_row["states"][step - 1])
    partner = int(cf_row["swapped_inputs"][1])
    changed = [(source_pos["table"][(fn, accessed)], cf_pos["table"][(fn, accessed)])]
    swap_partner = [(source_pos["table"][(fn, partner)], cf_pos["table"][(fn, partner)])]
    all_table = [(source_pos["table"][key], cf_pos["table"][key]) for key in sorted(source_pos["table"])]
    def function_table(function_index):
        return [
            (source_pos["table"][(function_index, value)],
             cf_pos["table"][(function_index, value)])
            for value in range(8)
        ]
    program_functions = list(map(int, source_row["program"]))
    program = list(zip(source_pos["program"], cf_pos["program"]))
    start = [(source_pos["start"], cf_pos["start"])]
    answer = [(int(source_answer), int(cf_answer))]
    step_tables = {
        f"step{index + 1}_table": function_table(function_index)
        for index, function_index in enumerate(program_functions)
    }
    result = {
        "changed_cell": changed,
        "swap_partner": swap_partner,
        "both_changed": changed + swap_partner,
        "all_table": all_table,
        "changed_function_table": function_table(fn),
        "final_function_table": function_table(program_functions[-1]),
        "program": program,
        "start": start,
        "answer": answer,
        "all_semantic": all_table + program + start + answer,
    }
    result.update(step_tables)
    return result


def capture_values(model, model_input, t, sc, block, use_bf16, component="v"):
    attention = model.blocks[block - 1].attn
    captured = {}

    def hook(_module, _inputs, output):
        bsz, length, _ = output.shape
        reshaped = output.reshape(
            bsz, length, 3, attention.num_heads,
            attention.dim // attention.num_heads,
        )
        component_index = {"k": 1, "v": 2}[component]
        captured["values"] = reshaped[:, :, component_index].detach().clone()

    handle = attention.qkv.register_forward_hook(hook)
    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            endpoint, _ = model(model_input, t, deterministic=True,
                                self_cond_cfg_scale=sc, decoder_step_active=None)
    finally:
        handle.remove()
    return endpoint, captured["values"]


def patched_value_forward(model, model_input, t, sc, block, head,
                          donor_values, mappings, use_bf16, component="v"):
    attention = model.blocks[block - 1].attn
    prefix = model.num_model_mode_tokens + model.num_time_tokens + model.num_self_cond_cfg_tokens

    def hook(_module, _inputs, output):
        patched = output.clone()
        bsz, length, _ = patched.shape
        reshaped = patched.reshape(
            bsz, length, 3, attention.num_heads,
            attention.dim // attention.num_heads,
        )
        component_index = {"k": 1, "v": 2}[component]
        for row, pairs in enumerate(mappings):
            for source_position, donor_position in pairs:
                reshaped[row, source_position + prefix, component_index, head] = donor_values[
                    row, donor_position + prefix, head
                ].to(reshaped.dtype)
        return reshaped.reshape_as(output)

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
    subsets = {
        "all": rows,
        "edited_function_not_final": [r for r in rows if not r["edited_function_is_final"]],
    }
    for subset, selected in subsets.items():
        buckets = defaultdict(list)
        for row in selected:
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


def plot(metrics, output_dir, component="v"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in metrics if r["subset"] == "all"]
    matrix = np.asarray([
        [next(r["counterfactual_answer_rate"] for r in rows
              if r["intervention_step"] == step and r["mode"] == mode)
         for mode in MODES]
        for step in range(1, 5)
    ])
    fig, axis = plt.subplots(figsize=(11, 4.5))
    image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks(range(len(MODES)), MODES, rotation=30, ha="right")
    axis.set_yticks(range(4), ["edit s1", "edit s2", "edit s3", "edit s4"])
    axis.set_title(f"Block-11 head-3 {component.upper()} path patching")
    fig.colorbar(image, ax=axis, label="Follows counterfactual answer")
    fig.tight_layout(); fig.savefig(output_dir / f"head3_{component}_position_patching.png", dpi=180); plt.close(fig)


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
        source_endpoint, _ = capture_values(
            model, source["model_input"], source["t"], source["sc"], args.block, use_bf16,
            component=args.component,
        )
        cf_endpoint, donor_values = capture_values(
            model, cf["model_input"], cf["t"], cf["sc"], args.block, use_bf16,
            component=args.component,
        )
        source_stats, cf_stats = decode(source_endpoint, source, model, config), decode(cf_endpoint, cf, model, config)
        source_tokens = torch.tensor([token_map[int(x)] for x in source_batch["target"]], device=device)
        cf_tokens = torch.tensor([token_map[int(x)] for x in cf_batch["target"]], device=device)
        source_rows = [originals[index] for index in range(cursor, cursor + bsz)]
        cf_rows = [counterfactuals[index] for index in range(cursor, cursor + bsz)]
        mapping_by_mode = {mode: [] for mode in MODES}
        for local in range(bsz):
            mappings = position_mappings(
                source_rows[local], cf_rows[local], source["positions"][local],
                cf["positions"][local], tokenizer,
            )
            for mode in MODES: mapping_by_mode[mode].append(mappings[mode])
        for mode in MODES:
            endpoint = patched_value_forward(
                model, source["model_input"], source["t"], source["sc"],
                args.block, args.head, donor_values, mapping_by_mode[mode], use_bf16,
                component=args.component,
            )
            stats = decode(endpoint, source, model, config)
            for local in range(bsz):
                prediction = stats["prediction"][local]
                follows_source = bool(prediction == source_tokens[local])
                follows_cf = bool(prediction == cf_tokens[local])
                rows.append({
                    "pair_index": cursor + local,
                    "intervention_step": int(cf_rows[local]["intervention_step"]),
                    "mode": mode, "follows_original": follows_source,
                    "edited_function_is_final": bool(
                        int(source_rows[local]["program"][
                            int(cf_rows[local]["intervention_step"]) - 1
                        ]) == int(source_rows[local]["program"][3])
                    ),
                    "follows_counterfactual": follows_cf,
                    "follows_other": not follows_source and not follows_cf,
                    "original_baseline_correct": bool(source_stats["correct"][local]),
                    "counterfactual_baseline_correct": bool(cf_stats["correct"][local]),
                })
        cursor += bsz; print(f"Patched {cursor}/{len(originals)} pairs", flush=True)
    metrics = aggregate(rows)
    summary = {
        "checkpoint": args.checkpoint, "checkpoint_step": checkpoint_step,
        "block": args.block, "head": args.head, "component": args.component,
        "counts": counts,
        "patch_definition": f"replace selected block-11 head-3 {args.component.upper()} vectors with semantic-position-aligned counterfactual vectors",
        "aggregate": metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "position_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    plot(metrics, output_dir, args.component)
    print(f"Saved head-position artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    run(parse_args())
