#!/usr/bin/env python
"""Functional equivalence tests for late-flow semantic caching in ELF.

This diagnostic does not yet save wall-clock compute: hooks replace already
computed head-3 K/V tensors.  It asks the prerequisite causal question—whether
late flow steps can reuse an earlier semantic state without changing answers.
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
from official_diagnostics import load_model
from official_flow_layer_causal_map import decode_state, ode_step, prepare_flow
from official_flow_trajectory import answer_decoder_stats
from official_layerwise_probe import select_balanced
from official_temporal_causal_tracing import capture_qkv
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split
from utils.generation_utils import _dlm_decode_logits
from utils.sampling_utils import get_sampling_steps


MODES = (
    "cache_k", "cache_v", "cache_kv", "zero_head3",
    "periodic2_kv", "periodic4_kv",
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
    parser.add_argument("--block", type=int, default=11)
    parser.add_argument("--head", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260921)
    return parser.parse_args()


def _conditional_hook_guard(cfg):
    return 1 if cfg == 1.0 else 2


def cached_qkv_step(model, state, cached_qkv, components, t, t_next,
                    config, cfg, self_cond_cfg, block, head):
    """Reuse cached text-position K/V on the conditional ODE call."""
    attention = model.blocks[block - 1].attn
    module = attention.qkv
    prefix = (
        model.num_model_mode_tokens + model.num_time_tokens
        + model.num_self_cond_cfg_tokens
    )
    indices = [{"k": 1, "v": 2}[component] for component in components]
    calls = 0

    def hook(_module, _inputs, output):
        nonlocal calls
        current = calls; calls += 1
        if current != 0:
            return output
        bsz, length, _ = output.shape
        patched = output.clone().reshape(
            bsz, length, 3, attention.num_heads,
            attention.dim // attention.num_heads,
        )
        cached = cached_qkv.reshape_as(patched)
        for index in indices:
            patched[:, prefix:, index, head] = cached[:, prefix:, index, head].to(
                patched.dtype
            )
        return patched.reshape_as(output)

    handle = module.register_forward_hook(hook)
    try:
        result = ode_step(model, state, t, t_next, config, cfg, self_cond_cfg)
    finally:
        handle.remove()
    expected = _conditional_hook_guard(cfg)
    if calls != expected:
        raise RuntimeError(f"Expected {expected} qkv calls, observed {calls}")
    return result


def native_capture_step(model, state, t, t_next, config, cfg, self_cond_cfg,
                        block):
    """Advance natively and capture conditional QKV without an extra forward."""
    module = model.blocks[block - 1].attn.qkv
    captured = {}
    calls = 0

    def hook(_module, _inputs, output):
        nonlocal calls
        current = calls; calls += 1
        if current == 0:
            captured["qkv"] = output.detach().clone()

    handle = module.register_forward_hook(hook)
    try:
        result = ode_step(model, state, t, t_next, config, cfg, self_cond_cfg)
    finally:
        handle.remove()
    expected = _conditional_hook_guard(cfg)
    if calls != expected or "qkv" not in captured:
        raise RuntimeError(f"Invalid capture: calls={calls}, captured={list(captured)}")
    return result, captured["qkv"]


def zero_head_step(model, state, t, t_next, config, cfg, self_cond_cfg,
                   block, head):
    """Zero one attention head before output projection on the conditional call."""
    attention = model.blocks[block - 1].attn
    calls = 0

    def hook(_module, inputs):
        nonlocal calls
        current = calls; calls += 1
        if current != 0:
            return None
        output = inputs[0].clone()
        shaped = output.reshape(
            output.shape[0], output.shape[1], attention.num_heads,
            attention.dim // attention.num_heads,
        )
        shaped[:, :, head] = 0
        return (shaped.reshape_as(output),)

    handle = attention.proj.register_forward_pre_hook(hook)
    try:
        result = ode_step(model, state, t, t_next, config, cfg, self_cond_cfg)
    finally:
        handle.remove()
    expected = _conditional_hook_guard(cfg)
    if calls != expected:
        raise RuntimeError(f"Expected {expected} projection calls, observed {calls}")
    return result


def run_mode(model, initial_state, initial_cache, start_index, t_steps, mode,
             config, args):
    state = dict(initial_state)
    cache = initial_cache
    for step_index in range(start_index, args.flow_steps):
        if mode == "cache_k":
            state = cached_qkv_step(
                model, state, cache, ("k",), t_steps[step_index],
                t_steps[step_index + 1], config, args.cfg,
                args.self_cond_cfg, args.block, args.head,
            )
        elif mode == "cache_v":
            state = cached_qkv_step(
                model, state, cache, ("v",), t_steps[step_index],
                t_steps[step_index + 1], config, args.cfg,
                args.self_cond_cfg, args.block, args.head,
            )
        elif mode == "cache_kv":
            state = cached_qkv_step(
                model, state, cache, ("k", "v"), t_steps[step_index],
                t_steps[step_index + 1], config, args.cfg,
                args.self_cond_cfg, args.block, args.head,
            )
        elif mode == "zero_head3":
            state = zero_head_step(
                model, state, t_steps[step_index], t_steps[step_index + 1],
                config, args.cfg, args.self_cond_cfg, args.block, args.head,
            )
        elif mode.startswith("periodic"):
            interval = 2 if mode == "periodic2_kv" else 4
            if (step_index - start_index) % interval == 0:
                state, cache = native_capture_step(
                    model, state, t_steps[step_index], t_steps[step_index + 1],
                    config, args.cfg, args.self_cond_cfg, args.block,
                )
            else:
                state = cached_qkv_step(
                    model, state, cache, ("k", "v"), t_steps[step_index],
                    t_steps[step_index + 1], config, args.cfg,
                    args.self_cond_cfg, args.block, args.head,
                )
        else:
            raise ValueError(mode)
    return state


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
            "native_answer_agreement": float(np.mean([r["native_answer_agreement"] for r in items])),
            "answer_state_l2": float(np.mean([r["answer_state_l2"] for r in items])),
            "native_exact_accuracy": float(np.mean([r["native_exact_correct"] for r in items])),
        })
    return metrics


def plot(metrics, output_dir, cutoffs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = ["compose/d1", "compose/d2", "compose/d4", "lookup/d4"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for axis, group in zip(axes.flat, groups):
        chosen = [m for m in metrics if m["group"] == group]
        native = chosen[0]["native_exact_accuracy"] if chosen else np.nan
        axis.axhline(native, color="black", linestyle="--", linewidth=1, label="native")
        for mode in MODES:
            rows = sorted([m for m in chosen if m["mode"] == mode], key=lambda x: x["cutoff_time"])
            axis.plot([m["cutoff_time"] for m in rows],
                      [m["exact_accuracy"] for m in rows], marker="o", label=mode)
        axis.set_title(group); axis.set_ylim(-.03, 1.03); axis.grid(alpha=.25)
        axis.set_xticks(cutoffs)
    for axis in axes[-1]: axis.set_xlabel("Cache/ablation cutoff time")
    for axis in axes[:, 0]: axis.set_ylabel("Exact answer accuracy")
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "semantic_cache_accuracy.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for axis, group in zip(axes.flat, groups):
        chosen = [m for m in metrics if m["group"] == group]
        for mode in MODES:
            rows = sorted([m for m in chosen if m["mode"] == mode], key=lambda x: x["cutoff_time"])
            axis.plot([m["cutoff_time"] for m in rows],
                      [m["native_answer_agreement"] for m in rows], marker="o", label=mode)
        axis.set_title(group); axis.set_ylim(-.03, 1.03); axis.grid(alpha=.25)
        axis.set_xticks(cutoffs)
    for axis in axes[-1]: axis.set_xlabel("Cache/ablation cutoff time")
    for axis in axes[:, 0]: axis.set_ylabel("Native answer-token agreement")
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "semantic_cache_agreement.png", dpi=220)
    plt.close(fig)


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
            raise ValueError(f"Cutoff {cutoff} is not on schedule {t_steps.tolist()}")
        cutoff_indices.append(index)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    use_bf16 = bool(config.use_bf16) and device.type == "cuda"
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
        qkv = []
        for step_index in range(args.flow_steps):
            qkv.append(capture_qkv(
                model, state, t_steps[step_index], args.block,
                args.self_cond_cfg, use_bf16,
            ))
            state = ode_step(
                model, state, t_steps[step_index], t_steps[step_index + 1],
                config, args.cfg, args.self_cond_cfg,
            )
            states.append(state)
        native_state = state
        native_stats = decode_state(native_state, model, config, args.self_cond_cfg)
        native_prediction = native_stats["prediction"]
        answer_positions = states[0]["positions"]
        batch_rows = torch.arange(bsz, device=device)

        for cutoff, cutoff_index in zip(cutoffs, cutoff_indices):
            for mode in MODES:
                final_state = run_mode(
                    model, states[cutoff_index], qkv[cutoff_index], cutoff_index,
                    t_steps, mode, config, args,
                )
                stats = decode_state(final_state, model, config, args.self_cond_cfg)
                delta = (
                    final_state["z"][batch_rows, answer_positions].float()
                    - native_state["z"][batch_rows, answer_positions].float()
                ).norm(dim=-1)
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
                    })
        cursor += bsz
        print(f"Semantic cache: processed {cursor}/{len(selected)} samples", flush=True)

    metrics = aggregate(rows)
    summary = {
        "checkpoint": args.checkpoint, "checkpoint_step": checkpoint_step,
        "device": str(device), "seed": args.seed,
        "samples_per_group": args.samples_per_group,
        "flow_steps": args.flow_steps, "schedule": "uniform",
        "t_steps": [float(x) for x in t_steps], "cutoffs": cutoffs,
        "block": args.block, "head": args.head, "modes": list(MODES),
        "cache_scope": "head-3 K/V at text positions only; model/time/self-cond prefix remains native",
        "warning": "functional replacement diagnostic only; hooks do not yet reduce FLOPs or latency",
        "source_ids": source_ids, "metrics": metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "cache_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader(); writer.writerows(metrics)
    with (output_dir / "per_sample.jsonl").open("w") as handle:
        for row in rows: handle.write(json.dumps(row) + "\n")
    plot(metrics, output_dir, cutoffs)
    print(f"Saved semantic-cache artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    run(parse_args())
