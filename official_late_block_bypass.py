#!/usr/bin/env python
"""Test identity-bypassing late ELF blocks after semantic commitment."""

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
from official_diagnostics import load_model
from official_flow_layer_causal_map import decode_state, ode_step, prepare_flow
from official_layerwise_probe import select_balanced
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split
from utils.sampling_utils import get_sampling_steps


BLOCK_SETS = {
    "b11": (11,),
    "b10_11": (10, 11),
    "b9_11": (9, 10, 11),
    "b8_11": (8, 9, 10, 11),
}
MODES = tuple(
    f"{scope}_{name}"
    for scope in ("cond", "all")
    for name in BLOCK_SETS
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-group", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--cutoff", action="append", type=float, default=None)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--self-cond-cfg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260921)
    return parser.parse_args()


def parse_mode(mode):
    scope, name = mode.split("_", 1)
    return scope, BLOCK_SETS[name]


def bypass_step(model, state, t, t_next, config, cfg, self_cond_cfg,
                block_numbers, scope):
    """Return selected residual blocks' inputs on conditional or all CFG calls."""
    calls = {block: 0 for block in block_numbers}
    handles = []

    for block_number in block_numbers:
        module = model.blocks[block_number - 1]

        def hook(_module, inputs, output, block_number=block_number):
            current = calls[block_number]
            calls[block_number] += 1
            if scope == "all" or current == 0:
                return inputs[0]
            return output

        handles.append(module.register_forward_hook(hook))
    try:
        result = ode_step(model, state, t, t_next, config, cfg, self_cond_cfg)
    finally:
        for handle in handles:
            handle.remove()
    expected = 1 if cfg == 1.0 else 2
    invalid = {block: count for block, count in calls.items() if count != expected}
    if invalid:
        raise RuntimeError(f"Unexpected block-hook calls: {invalid}; expected {expected}")
    return result


def aggregate(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["group"], row["mode"], row["cutoff_time"])].append(row)
    metrics = []
    for (group, mode, cutoff), items in sorted(buckets.items()):
        metrics.append({
            "group": group, "mode": mode, "cutoff_time": cutoff,
            "samples": len(items),
            "exact_accuracy": float(np.mean([r["exact_correct"] for r in items])),
            "native_answer_agreement": float(np.mean([
                r["native_answer_agreement"] for r in items
            ])),
            "answer_state_l2": float(np.mean([r["answer_state_l2"] for r in items])),
            "native_exact_accuracy": float(np.mean([
                r["native_exact_correct"] for r in items
            ])),
            "theoretical_block_call_reduction": items[0]["theoretical_block_call_reduction"],
        })
    return metrics


def plot(metrics, output_dir, cutoffs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = ["compose/d1", "compose/d2", "compose/d4", "lookup/d4"]
    for metric, ylabel, filename in (
        ("exact_accuracy", "Exact answer accuracy", "late_block_bypass_accuracy.png"),
        ("native_answer_agreement", "Native answer-token agreement", "late_block_bypass_agreement.png"),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
        for axis, group in zip(axes.flat, groups):
            chosen = [m for m in metrics if m["group"] == group]
            if metric == "exact_accuracy" and chosen:
                axis.axhline(chosen[0]["native_exact_accuracy"], color="black",
                             linestyle="--", linewidth=1, label="native")
            for mode in MODES:
                values = sorted(
                    [m for m in chosen if m["mode"] == mode],
                    key=lambda x: x["cutoff_time"],
                )
                axis.plot([m["cutoff_time"] for m in values],
                          [m[metric] for m in values], marker="o", label=mode)
            axis.set_title(group); axis.set_ylim(-.03, 1.03); axis.grid(alpha=.25)
            axis.set_xticks(cutoffs)
        for axis in axes[-1]: axis.set_xlabel("Bypass cutoff time")
        for axis in axes[:, 0]: axis.set_ylabel(ylabel)
        axes[0, 0].legend(frameon=False, fontsize=7, ncol=3)
        fig.tight_layout(); fig.savefig(output_dir / filename, dpi=220); plt.close(fig)


@torch.no_grad()
def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.samples_per_group % args.batch_size:
        raise ValueError("--batch-size must divide --samples-per-group")
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
    t_steps = get_sampling_steps(
        args.flow_steps, "uniform", config.denoiser_p_mean,
        config.denoiser_p_std, device=device, dtype=dtype,
    )
    cutoffs = args.cutoff or [0.25, 0.5, 0.625]
    cutoff_indices = []
    for cutoff in cutoffs:
        index = int(torch.argmin(torch.abs(t_steps.float() - cutoff)))
        if abs(float(t_steps[index]) - cutoff) > 1e-5:
            raise ValueError(f"Cutoff {cutoff} is not on schedule")
        cutoff_indices.append(index)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    rows = []
    cursor = 0

    for batch in loader:
        bsz = len(batch["target"])
        groups = expected_groups[cursor:cursor + bsz]
        if len(set(groups)) != 1:
            raise ValueError(f"Batch crosses groups: {set(groups)}")
        noise = torch.randn(
            (bsz, config.max_length, model.text_encoder_dim),
            generator=generator, dtype=dtype,
        )
        state = prepare_flow(batch, noise, model, encoder, tokenizer, config, device)
        states = [state]
        for step_index in range(args.flow_steps):
            state = ode_step(
                model, state, t_steps[step_index], t_steps[step_index + 1],
                config, args.cfg, args.self_cond_cfg,
            )
            states.append(state)
        native_state = state
        native_stats = decode_state(native_state, model, config, args.self_cond_cfg)
        native_prediction = native_stats["prediction"]
        positions = states[0]["positions"]
        batch_rows = torch.arange(bsz, device=device)

        for cutoff, cutoff_index in zip(cutoffs, cutoff_indices):
            late_steps = args.flow_steps - cutoff_index
            for mode in MODES:
                scope, blocks = parse_mode(mode)
                final_state = dict(states[cutoff_index])
                for step_index in range(cutoff_index, args.flow_steps):
                    final_state = bypass_step(
                        model, final_state, t_steps[step_index],
                        t_steps[step_index + 1], config, args.cfg,
                        args.self_cond_cfg, blocks, scope,
                    )
                stats = decode_state(final_state, model, config, args.self_cond_cfg)
                delta = (
                    final_state["z"][batch_rows, positions].float()
                    - native_state["z"][batch_rows, positions].float()
                ).norm(dim=-1)
                cfg_calls = 1 if args.cfg == 1.0 else 2
                skipped_calls = late_steps * len(blocks) * (
                    1 if scope == "cond" else cfg_calls
                )
                total_calls = args.flow_steps * model.depth * cfg_calls
                reduction = skipped_calls / total_calls
                for local in range(bsz):
                    rows.append({
                        "source_id": source_ids[cursor + local],
                        "group": groups[local], "mode": mode,
                        "cutoff_time": cutoff, "cutoff_index": cutoff_index,
                        "exact_correct": bool(stats["correct"][local]),
                        "native_exact_correct": bool(native_stats["correct"][local]),
                        "native_answer_agreement": bool(
                            stats["prediction"][local] == native_prediction[local]
                        ),
                        "answer_state_l2": float(delta[local]),
                        "theoretical_block_call_reduction": reduction,
                    })
        cursor += bsz
        print(f"Late-block bypass: processed {cursor}/{len(selected)} samples", flush=True)

    metrics = aggregate(rows)
    summary = {
        "checkpoint": args.checkpoint, "checkpoint_step": checkpoint_step,
        "device": str(device), "seed": args.seed,
        "samples_per_group": args.samples_per_group,
        "flow_steps": args.flow_steps, "schedule": "uniform",
        "t_steps": [float(x) for x in t_steps], "cutoffs": cutoffs,
        "modes": list(MODES), "block_sets": BLOCK_SETS,
        "bypass_definition": "selected ELF residual blocks return their input unchanged",
        "warning": "hook diagnostic computes blocks before replacing outputs; theoretical reduction assumes true conditional execution",
        "source_ids": source_ids, "metrics": metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "bypass_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader(); writer.writerows(metrics)
    with (output_dir / "per_sample.jsonl").open("w") as handle:
        for row in rows: handle.write(json.dumps(row) + "\n")
    plot(metrics, output_dir, cutoffs)
    print(f"Saved late-block bypass artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    run(parse_args())
