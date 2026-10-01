#!/usr/bin/env python
"""Causal patching with minimal function-table counterfactual pairs."""

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "official-elf" / "src"))

from configs.config import load_config_from_yaml
from make_official_composition import render
from modules.t5_encoder import get_encoder
from official_component_patching import capture_components, patched_component_forward
from official_diagnostics import load_model
from official_flow_trajectory import answer_decoder_stats
from official_intermediate_state_probe import select_compose_d4
from official_layerwise_patching import (
    _capture_first_forward, _patched_forward, semantic_token_map,
)
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split
from utils.encoder_utils import encode_text
from utils.generation_utils import _dlm_decode_logits
from utils.sampling_utils import restore_cond


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pairs-per-step", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260919)
    return parser.parse_args()


def parse_tables(prompt):
    tables = [None] * 4
    for match in re.finditer(r"F([0-3]): ([0-7](?: [0-7]){7})", prompt):
        tables[int(match.group(1))] = list(map(int, match.group(2).split()))
    if any(table is None for table in tables):
        raise ValueError(f"Could not parse four tables from: {prompt}")
    return tables


def execute(tables, start, program):
    states = [int(start)]
    for fn in program:
        states.append(int(tables[int(fn)][states[-1]]))
    return states


def counterfactual_for_step(row, step, tokenizer, rng):
    """Swap one accessed table output; preserve all states before `step`."""
    tables = parse_tables(str(row["input"]))
    program = list(map(int, row["program"]))
    original_states = list(map(int, row["states"]))
    fn = program[step - 1]
    accessed_input = original_states[step - 1]
    candidates = list(range(8))
    rng.shuffle(candidates)
    for alternate_input in candidates:
        if alternate_input == accessed_input:
            continue
        modified = [table.copy() for table in tables]
        modified[fn][accessed_input], modified[fn][alternate_input] = (
            modified[fn][alternate_input], modified[fn][accessed_input]
        )
        new_states = execute(modified, original_states[0], program)
        if new_states[:step] != original_states[:step]:
            continue
        if new_states[step] == original_states[step] or new_states[-1] == original_states[-1]:
            continue
        changed = {
            "tables": modified, "program": program, "task": "compose",
            "start": original_states[0],
        }
        prompt = render(changed)
        return {
            "condition_input_ids": tokenizer(prompt, add_special_tokens=False)["input_ids"],
            "input_ids": tokenizer(str(new_states[-1]), add_special_tokens=False)["input_ids"],
            "input": prompt, "target": str(new_states[-1]), "task": "compose",
            "depth": len(program), "episode_id": f"{row['episode_id']}-cf-s{step}",
            "table_id": f"{row['table_id']}-cf-s{step}", "start": original_states[0],
            "program": program, "states": new_states,
            "intervention_step": step, "swapped_inputs": [accessed_input, alternate_input],
        }
    return None


def plain_row(row, step):
    result = {}
    for key, value in row.items():
        result[key] = value.tolist() if isinstance(value, np.ndarray) else value
    result["intervention_step"] = step
    return result


def build_pairs_for_depth(dataset, tokenizer, pairs_per_step, depth, seed):
    candidates = [row for row in dataset
                  if row["task"] == "compose" and int(row["depth"]) == depth]
    rng = random.Random(seed)
    originals, counterfactuals = [], []
    counts = defaultdict(int)
    for step in range(1, depth + 1):
        order = list(range(len(candidates))); rng.shuffle(order)
        for index in order:
            cf = counterfactual_for_step(candidates[index], step, tokenizer, rng)
            if cf is None:
                continue
            originals.append(plain_row(candidates[index], step))
            counterfactuals.append(cf)
            counts[step] += 1
            if counts[step] >= pairs_per_step:
                break
        if counts[step] < pairs_per_step:
            raise ValueError(f"Only built {counts[step]} pairs for intervention step {step}")
    return Dataset.from_list(originals), Dataset.from_list(counterfactuals), dict(counts)


def build_pairs(dataset, tokenizer, pairs_per_step, seed):
    """Backward-compatible d4 counterfactual builder."""
    return build_pairs_for_depth(dataset, tokenizer, pairs_per_step, 4, seed)


def prepare(batch, model, encoder, tokenizer, config, device, shared_noise):
    dtype = next(model.parameters()).dtype
    input_ids = torch.from_numpy(np.asarray(batch["input_ids"])).to(device).long()
    encoder_mask = torch.from_numpy(np.asarray(batch["encoder_attention_mask"])).to(device).float()
    cond_mask = torch.from_numpy(np.asarray(batch["cond_seq_mask"])).to(device).float()
    starts = cond_mask.long().sum(dim=1)
    lengths = torch.tensor(
        [len(tokenizer.encode(str(target), add_special_tokens=False)) for target in batch["target"]],
        device=device, dtype=torch.long,
    )
    positions = starts + lengths - 1
    cond_seq = encode_text(input_ids, encoder_mask, encoder, config.latent_mean,
                           config.latent_std, use_bf16=bool(config.use_bf16)).to(dtype)
    z = (shared_noise.to(device=device, dtype=dtype) * config.denoiser_noise_scale)
    z = restore_cond(z, cond_seq, cond_mask)
    previous = restore_cond(torch.zeros_like(z), cond_seq, cond_mask)
    return {
        "input_ids": input_ids, "starts": starts, "lengths": lengths,
        "positions": positions, "model_input": torch.cat([z, previous], dim=-1),
        "t": torch.zeros((len(batch["target"]),), device=device, dtype=dtype),
        "sc": torch.ones((len(batch["target"]),), device=device, dtype=dtype),
    }


def decode(endpoint, prepared, model, config):
    logits = _dlm_decode_logits(endpoint, model, 1.0, config, 1.0)
    return answer_decoder_stats(
        logits, prepared["starts"], prepared["lengths"], prepared["input_ids"],
    )


def aggregate(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["intervention_step"], row["patch_type"], row["location"])].append(row)
    result = []
    for (step, patch_type, location), items in sorted(buckets.items()):
        result.append({
            "intervention_step": step, "patch_type": patch_type, "location": location,
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
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for step in range(1, 5):
        chosen = [r for r in metrics if r["intervention_step"] == step and r["patch_type"] == "block"]
        axes[0].plot([int(r["location"].split("_")[1]) for r in chosen],
                     [r["counterfactual_answer_rate"] for r in chosen], marker="o", label=f"edit s{step}")
    axes[0].set_xlabel("Patched representation layer"); axes[0].set_ylabel("Follows counterfactual")
    axes[0].set_xticks(range(13)); axes[0].set_ylim(-0.03, 1.03); axes[0].grid(alpha=.25); axes[0].legend()
    component_locations = [
        "b9_attention", "b9_mlp", "b10_attention", "b10_mlp",
        "b11_attention", "b11_mlp",
    ]
    labels = ["B9-attn", "B9-mlp", "B10-attn", "B10-mlp", "B11-attn", "B11-mlp"]
    for step in range(1, 5):
        chosen = [r for r in metrics if r["intervention_step"] == step and r["patch_type"] == "component"]
        mapping = {r["location"]: r["counterfactual_answer_rate"] for r in chosen}
        axes[1].plot(range(len(labels)), [mapping[x] for x in component_locations], marker="o", label=f"edit s{step}")
    axes[1].set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    axes[1].set_ylabel("Follows counterfactual"); axes[1].set_ylim(-0.03, 1.03); axes[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(output_dir / "counterfactual_patching.png", dpi=180); plt.close(fig)


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
    rows = []
    cursor = 0
    for source_batch, cf_batch in zip(original_loader, counterfactual_loader):
        bsz = len(source_batch["target"])
        noise = torch.randn((bsz, config.max_length, model.text_encoder_dim), generator=generator, dtype=dtype)
        source = prepare(source_batch, model, encoder, tokenizer, config, device, noise)
        cf = prepare(cf_batch, model, encoder, tokenizer, config, device, noise)
        source_endpoint, _ = _capture_first_forward(
            model, source["model_input"], source["t"], source["sc"], source["positions"], use_bf16,
        )
        cf_endpoint, cf_hidden = _capture_first_forward(
            model, cf["model_input"], cf["t"], cf["sc"], cf["positions"], use_bf16,
        )
        _, cf_components = capture_components(
            model, cf["model_input"], cf["t"], cf["sc"], cf["positions"], [9, 10, 11], use_bf16,
        )
        source_stats, cf_stats = decode(source_endpoint, source, model, config), decode(cf_endpoint, cf, model, config)
        source_tokens = torch.tensor([token_map[int(x)] for x in source_batch["target"]], device=device)
        cf_tokens = torch.tensor([token_map[int(x)] for x in cf_batch["target"]], device=device)
        steps = list(map(int, originals[cursor:cursor + bsz]["intervention_step"]))

        interventions = []
        for layer in range(model.depth + 1):
            endpoint = _patched_forward(
                model, source["model_input"], source["t"], source["sc"], source["positions"],
                cf_hidden[:, layer], layer, 1.0, use_bf16,
            )
            interventions.append(("block", f"layer_{layer}", decode(endpoint, source, model, config)))
        for block in (9, 10, 11):
            for component in ("attention", "mlp"):
                endpoint = patched_component_forward(
                    model, source["model_input"], source["t"], source["sc"], source["positions"],
                    block, component, cf_components[(block, component)], use_bf16,
                )
                interventions.append(("component", f"b{block}_{component}", decode(endpoint, source, model, config)))

        for patch_type, location, stats in interventions:
            prediction = stats["prediction"]
            for local in range(bsz):
                follows_original = bool(prediction[local] == source_tokens[local])
                follows_cf = bool(prediction[local] == cf_tokens[local])
                rows.append({
                    "pair_index": cursor + local, "intervention_step": steps[local],
                    "patch_type": patch_type, "location": location,
                    "original_answer": int(source_batch["target"][local]),
                    "counterfactual_answer": int(cf_batch["target"][local]),
                    "follows_original": follows_original, "follows_counterfactual": follows_cf,
                    "follows_other": not follows_original and not follows_cf,
                    "original_baseline_correct": bool(source_stats["correct"][local]),
                    "counterfactual_baseline_correct": bool(cf_stats["correct"][local]),
                })
        cursor += bsz
        print(f"Patched {cursor}/{len(originals)} pairs", flush=True)
    metrics = aggregate(rows)
    summary = {
        "checkpoint": args.checkpoint, "checkpoint_step": checkpoint_step,
        "device": str(device), "counts": counts, "seed": args.seed,
        "pair_definition": "one within-permutation table-output swap; states before intervention unchanged; final answer changed",
        "aggregate": metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "counterfactual_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    with (output_dir / "per_sample.jsonl").open("w") as handle:
        for row in rows: handle.write(json.dumps(row) + "\n")
    plot(metrics, output_dir)
    print(f"Saved counterfactual patching artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    run(parse_args())
