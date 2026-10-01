#!/usr/bin/env python
"""Temporal causal tracing of ELF block-11 head-3 K/V across flow steps.

For a minimal source/counterfactual pair, inject donor K or V at flow step i,
optionally restore the clean source component at a later step j, finish the
native flow, and measure the final answer.  The difference between injection
only and injection+restoration estimates temporal causal mediation.
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
from official_counterfactual_patching import build_pairs
from official_diagnostics import load_model
from official_flow_layer_causal_map import (
    decode_state, ode_step, prepare_flow,
)
from official_layerwise_patching import semantic_token_map
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split
from utils.sampling_utils import get_sampling_steps


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pairs-per-step", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--self-cond-cfg", type=float, default=1.0)
    parser.add_argument("--block", type=int, default=11)
    parser.add_argument("--head", type=int, default=3)
    parser.add_argument("--component", action="append", choices=("k", "v"), default=None)
    parser.add_argument("--seed", type=int, default=20260921)
    return parser.parse_args()


def capture_qkv(model, state, t, block, self_cond_cfg, use_bf16):
    module = model.blocks[block - 1].attn.qkv
    captured = {}
    bsz = state["z"].shape[0]
    dtype = state["z"].dtype
    model_input = torch.cat([state["z"], state["previous"]], dim=-1)
    t_batch = torch.full((bsz,), float(t), device=state["z"].device, dtype=dtype)
    sc_batch = torch.full(
        (bsz,), float(self_cond_cfg), device=state["z"].device, dtype=dtype,
    )

    def hook(_module, _inputs, output):
        captured["qkv"] = output.detach().clone()

    handle = module.register_forward_hook(hook)
    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            model(
                model_input, t_batch, deterministic=True,
                self_cond_cfg_scale=sc_batch, decoder_step_active=None,
            )
    finally:
        handle.remove()
    return captured["qkv"]


def patched_component_step(model, state, replacement_qkv, component, t, t_next,
                           config, cfg, self_cond_cfg, block, head):
    """Replace one head's K or V on the conditional call of one ODE step."""
    attention = model.blocks[block - 1].attn
    module = attention.qkv
    component_index = {"k": 1, "v": 2}[component]
    calls = 0

    def hook(_module, _inputs, output):
        nonlocal calls
        current_call = calls
        calls += 1
        if current_call != 0:
            return output
        bsz, length, _ = output.shape
        patched = output.clone().reshape(
            bsz, length, 3, attention.num_heads,
            attention.dim // attention.num_heads,
        )
        replacement = replacement_qkv.reshape_as(patched)
        patched[:, :, component_index, head] = replacement[
            :, :, component_index, head
        ].to(patched.dtype)
        return patched.reshape_as(output)

    handle = module.register_forward_hook(hook)
    try:
        result = ode_step(
            model, state, t, t_next, config, cfg, self_cond_cfg,
        )
    finally:
        handle.remove()
    expected = 1 if cfg == 1.0 else 2
    if calls != expected:
        raise RuntimeError(f"Expected {expected} QKV-hook calls, observed {calls}")
    return result


def finish_from(model, initial_state, start_index, t_steps, config, cfg,
                self_cond_cfg, interventions, block, head):
    state = dict(initial_state)
    for step_index in range(start_index, len(t_steps) - 1):
        intervention = interventions.get(step_index)
        if intervention is None:
            state = ode_step(
                model, state, t_steps[step_index], t_steps[step_index + 1],
                config, cfg, self_cond_cfg,
            )
        else:
            component, replacement = intervention
            state = patched_component_step(
                model, state, replacement, component,
                t_steps[step_index], t_steps[step_index + 1],
                config, cfg, self_cond_cfg, block, head,
            )
    return state


def aggregate(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[(
            row["component"], row["intervention_step"],
            row["injection_index"], row["restoration_index"],
        )].append(row)
    result = []
    for key, items in sorted(buckets.items()):
        component, intervention_step, injection_index, restoration_index = key
        eligible = [item for item in items if item["eligible"]]
        selected = eligible or items
        result.append({
            "component": component,
            "intervention_step": intervention_step,
            "injection_index": injection_index,
            "injection_time": items[0]["injection_time"],
            "restoration_index": restoration_index,
            "restoration_time": items[0]["restoration_time"],
            "samples": len(items), "eligible_samples": len(eligible),
            "counterfactual_answer_rate": float(np.mean([
                item["follows_counterfactual"] for item in selected
            ])),
            "original_answer_rate": float(np.mean([
                item["follows_original"] for item in selected
            ])),
            "other_answer_rate": float(np.mean([
                item["follows_other"] for item in selected
            ])),
        })
    return result


def plot(rows, output_dir, components, flow_steps, t_steps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    relevant = {
        "k": [row for row in rows if row["component"] == "k"
              and row["intervention_step"] in (1, 2, 3) and row["eligible"]],
        "v": [row for row in rows if row["component"] == "v"
              and row["intervention_step"] == 4 and row["eligible"]],
    }
    fig, axes = plt.subplots(len(components), 2, figsize=(13, 5 * len(components)), squeeze=False)
    for row_index, component in enumerate(components):
        selected = relevant[component]
        pulse = np.full(flow_steps, np.nan)
        rescue = np.full((flow_steps, flow_steps), np.nan)
        restored_rate = np.full((flow_steps, flow_steps), np.nan)
        for injection in range(flow_steps):
            base = [row for row in selected if row["injection_index"] == injection
                    and row["restoration_index"] == -1]
            if base:
                pulse[injection] = np.mean([row["follows_counterfactual"] for row in base])
            for restoration in range(injection + 1, flow_steps):
                restored = [row for row in selected if row["injection_index"] == injection
                            and row["restoration_index"] == restoration]
                if restored:
                    rate = np.mean([row["follows_counterfactual"] for row in restored])
                    restored_rate[restoration, injection] = rate
                    rescue[restoration, injection] = pulse[injection] - rate
        axis = axes[row_index, 0]
        axis.plot(range(flow_steps), pulse, marker="o")
        axis.set_xticks(range(flow_steps), [f"{float(t_steps[i]):.2f}" for i in range(flow_steps)])
        axis.set_ylim(-.03, 1.03); axis.grid(alpha=.25)
        axis.set_xlabel("Donor-injection flow time")
        axis.set_ylabel("Final donor-answer rate")
        label = "edits s1–s3" if component == "k" else "edit s4"
        axis.set_title(f"Head-3 {component.upper()} pulse ({label})")

        axis = axes[row_index, 1]
        image = axis.imshow(rescue, origin="lower", aspect="auto", vmin=-1, vmax=1,
                            cmap="coolwarm")
        axis.set_xticks(range(flow_steps), [f"{float(t_steps[i]):.2f}" for i in range(flow_steps)])
        axis.set_yticks(range(flow_steps), [f"{float(t_steps[i]):.2f}" for i in range(flow_steps)])
        axis.set_xlabel("Donor-injection time")
        axis.set_ylabel("Source-restoration time")
        axis.set_title(f"Causal rescue: pulse rate − restored rate ({component.upper()})")
        fig.colorbar(image, ax=axis, label="Reduction in donor-answer rate")
    fig.tight_layout()
    fig.savefig(output_dir / "temporal_causal_tracing.png", dpi=220)
    plt.close(fig)


@torch.no_grad()
def run(args):
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
    validation = load_dataset_split(config.eval_data_path)
    originals, counterfactuals, counts = build_pairs(
        validation, tokenizer, args.pairs_per_step, args.seed,
    )
    loader_args = dict(
        batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False,
        max_seq_length=config.max_length,
        pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
        max_input_seq_length=config.max_input_length, distributed=False,
    )
    source_loader = get_dataloader(originals, **loader_args)
    donor_loader = get_dataloader(counterfactuals, **loader_args)
    dtype = next(model.parameters()).dtype
    t_steps = get_sampling_steps(
        args.flow_steps, "uniform", config.denoiser_p_mean,
        config.denoiser_p_std, device=device, dtype=dtype,
    )
    components = args.component or ["k", "v"]
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    use_bf16 = bool(config.use_bf16) and device.type == "cuda"
    rows = []
    cursor = 0

    for source_batch, donor_batch in zip(source_loader, donor_loader):
        bsz = len(source_batch["target"])
        noise = torch.randn(
            (bsz, config.max_length, model.text_encoder_dim),
            generator=generator, dtype=dtype,
        )
        source = prepare_flow(
            source_batch, noise, model, encoder, tokenizer, config, device,
        )
        donor = prepare_flow(
            donor_batch, noise, model, encoder, tokenizer, config, device,
        )
        source_states = [source]
        source_qkv, donor_qkv = [], []
        for step_index in range(args.flow_steps):
            source_qkv.append(capture_qkv(
                model, source, t_steps[step_index], args.block,
                args.self_cond_cfg, use_bf16,
            ))
            donor_qkv.append(capture_qkv(
                model, donor, t_steps[step_index], args.block,
                args.self_cond_cfg, use_bf16,
            ))
            source = ode_step(
                model, source, t_steps[step_index], t_steps[step_index + 1],
                config, args.cfg, args.self_cond_cfg,
            )
            donor = ode_step(
                model, donor, t_steps[step_index], t_steps[step_index + 1],
                config, args.cfg, args.self_cond_cfg,
            )
            source_states.append(source)
        source_baseline = decode_state(source, model, config, args.self_cond_cfg)
        donor_baseline = decode_state(donor, model, config, args.self_cond_cfg)
        source_tokens = torch.tensor(
            [token_map[int(x)] for x in source_batch["target"]], device=device,
        )
        donor_tokens = torch.tensor(
            [token_map[int(x)] for x in donor_batch["target"]], device=device,
        )
        edit_steps = list(map(
            int, originals[cursor:cursor + bsz]["intervention_step"],
        ))

        for component in components:
            for injection_index in range(args.flow_steps):
                conditions = [(-1, {
                    injection_index: (component, donor_qkv[injection_index]),
                })]
                for restoration_index in range(injection_index + 1, args.flow_steps):
                    conditions.append((restoration_index, {
                        injection_index: (component, donor_qkv[injection_index]),
                        restoration_index: (component, source_qkv[restoration_index]),
                    }))
                for restoration_index, interventions in conditions:
                    final_state = finish_from(
                        model, source_states[injection_index], injection_index,
                        t_steps, config, args.cfg, args.self_cond_cfg,
                        interventions, args.block, args.head,
                    )
                    stats = decode_state(
                        final_state, model, config, args.self_cond_cfg,
                    )
                    for local in range(bsz):
                        prediction = stats["prediction"][local]
                        follows_source = bool(prediction == source_tokens[local])
                        follows_donor = bool(prediction == donor_tokens[local])
                        rows.append({
                            "pair_index": cursor + local,
                            "intervention_step": edit_steps[local],
                            "component": component,
                            "injection_index": injection_index,
                            "injection_time": float(t_steps[injection_index]),
                            "restoration_index": restoration_index,
                            "restoration_time": (
                                None if restoration_index < 0
                                else float(t_steps[restoration_index])
                            ),
                            "follows_original": follows_source,
                            "follows_counterfactual": follows_donor,
                            "follows_other": not follows_source and not follows_donor,
                            "source_baseline_correct": bool(source_baseline["correct"][local]),
                            "donor_baseline_correct": bool(donor_baseline["correct"][local]),
                            "eligible": bool(
                                source_baseline["correct"][local]
                                and donor_baseline["correct"][local]
                            ),
                        })
        cursor += bsz
        print(f"Temporal trace: processed {cursor}/{len(originals)} pairs", flush=True)

    metrics = aggregate(rows)
    summary = {
        "checkpoint": args.checkpoint, "checkpoint_step": checkpoint_step,
        "device": str(device), "seed": args.seed, "counts": counts,
        "flow_steps": args.flow_steps, "schedule": "uniform",
        "t_steps": [float(x) for x in t_steps], "block": args.block,
        "head": args.head, "components": components,
        "injection": "replace all positions of block-11 head-3 K or V on one conditional ODE call with paired donor",
        "restoration": "at a later conditional ODE call, replace the same component with its clean source-baseline value",
        "relevant_subsets": {"k": "edits s1-s3", "v": "edit s4"},
        "source_baseline_accuracy": float(np.mean([
            row["source_baseline_correct"] for row in rows
        ])),
        "donor_baseline_accuracy": float(np.mean([
            row["donor_baseline_correct"] for row in rows
        ])),
        "aggregate": metrics,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    with (output_dir / "temporal_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader(); writer.writerows(metrics)
    with (output_dir / "per_sample.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    plot(rows, output_dir, components, args.flow_steps, t_steps)
    print(json.dumps({
        "source_baseline_accuracy": summary["source_baseline_accuracy"],
        "donor_baseline_accuracy": summary["donor_baseline_accuracy"],
        "rows": len(rows),
    }, indent=2), flush=True)
    print(f"Saved temporal causal trace to {output_dir}", flush=True)


if __name__ == "__main__":
    run(parse_args())
