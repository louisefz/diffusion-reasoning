#!/usr/bin/env python
"""Decompose solution gains across ELF transport and self-conditioning channels."""

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
from official_autonomous_correction import decode_correctness
from official_diagnostics import load_model
from official_flow_error_correction import clone_state, repeat_batch
from official_flow_layer_causal_map import prepare_flow
from official_reasoning_velocity_transport import advance, forward_velocity
from official_sudoku_error_injection import write_csv
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split
from utils.sampling_utils import get_sampling_steps, restore_cond


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--eval-data-path", default=None)
    p.add_argument("--difficulty", default="hard")
    p.add_argument("--num-samples", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--flow-steps", type=int, default=16)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--self-cond-cfg", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20261003)
    return p.parse_args()


def prediction_correctness(state, prediction, model, config, self_cond_cfg):
    predicted = dict(state)
    predicted["z"] = prediction
    return decode_correctness(predicted, model, config, self_cond_cfg)


def aggregate(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["mode"], row["call_index"])].append(row)
    result = []
    for (mode, call_index), items in sorted(buckets.items()):
        result.append({
            "mode": mode,
            "call_index": call_index,
            "cumulative_calls": call_index + 1,
            "flow_time": items[0]["flow_time"],
            "trajectories": len(items),
            "predicted_exact_accuracy": float(np.mean([
                x["predicted_exact"] for x in items])),
            "predicted_target_token_accuracy": float(np.mean([
                x["predicted_target_token_accuracy"] for x in items])),
            "wrong_to_correct": int(sum(x["wrong_to_correct"] for x in items)),
            "correct_to_wrong": int(sum(x["correct_to_wrong"] for x in items)),
        })
    return result


def make_plot(summary, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = ["native", "memory_zero", "memory_fixed_after_call1", "z_frozen_recurrence"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for mode in modes:
        curve = sorted([x for x in summary if x["mode"] == mode],
                       key=lambda x: x["call_index"])
        axes[0].plot([x["cumulative_calls"] for x in curve],
                     [x["predicted_exact_accuracy"] for x in curve],
                     marker="o", label=mode)
        axes[1].plot([x["cumulative_calls"] for x in curve],
                     [x["predicted_target_token_accuracy"] for x in curve],
                     marker="o", label=mode)
    axes[0].set(title="Whole-grid solution quality", ylabel="Exact accuracy")
    axes[1].set(title="Cell-level solution quality", ylabel="Target-token accuracy")
    for axis in axes:
        axis.set_xlabel("Cumulative denoiser calls")
        axis.set_ylim(-.03, 1.03)
        axis.grid(alpha=.25)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Where does iterative solution improvement come from?")
    fig.tight_layout()
    fig.savefig(output_dir / "compute_decomposition.png", dpi=220)
    plt.close(fig)


@torch.no_grad()
def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config_from_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    model, checkpoint_step = load_model(config, args.checkpoint, encoder_config, tokenizer, device)
    dataset = load_dataset_split(args.eval_data_path or config.eval_data_path)
    dataset = dataset.filter(lambda row: str(row["difficulty"]) == args.difficulty)
    dataset = dataset.select(range(min(args.num_samples, len(dataset))))
    loader = get_dataloader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        drop_last=False, max_seq_length=config.max_length,
        pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
        max_input_seq_length=config.max_input_length, distributed=False)
    dtype = next(model.parameters()).dtype
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    t_steps = get_sampling_steps(
        args.flow_steps, "uniform", config.denoiser_p_mean, config.denoiser_p_std,
        device=device, dtype=dtype)
    modes = ["native", "memory_zero", "memory_fixed_after_call1", "z_frozen_recurrence"]
    rows = []
    final_rows = []
    seen = 0
    for batch in loader:
        bsz = len(batch["target"])
        repeated = repeat_batch(batch, args.restarts)
        noise = torch.randn(
            (bsz * args.restarts, config.max_length, model.text_encoder_dim),
            generator=generator, dtype=dtype)
        initial = prepare_flow(repeated, noise, model, encoder, tokenizer, config, device)
        for mode in modes:
            state = clone_state(initial)
            fixed_memory = None
            previous_exact = None
            last_prediction = None
            for index in range(args.flow_steps):
                velocity, prediction = forward_velocity(
                    model, state, t_steps[index], config, args.cfg, args.self_cond_cfg)
                token_correct, mask, exact = prediction_correctness(
                    state, prediction, model, config, args.self_cond_cfg)
                if previous_exact is None:
                    wc = torch.zeros_like(exact, dtype=torch.bool)
                    cw = torch.zeros_like(exact, dtype=torch.bool)
                else:
                    wc = (~previous_exact) & exact
                    cw = previous_exact & (~exact)
                for local in range(len(exact)):
                    rows.append({
                        "example_index": seen + local // args.restarts,
                        "restart": local % args.restarts,
                        "mode": mode,
                        "call_index": index,
                        "flow_time": float(t_steps[index]),
                        "predicted_exact": bool(exact[local]),
                        "predicted_target_token_accuracy": float(
                            token_correct[local][mask[local]].float().mean()),
                        "wrong_to_correct": bool(wc[local]),
                        "correct_to_wrong": bool(cw[local]),
                    })
                previous_exact = exact
                last_prediction = prediction

                if mode == "z_frozen_recurrence":
                    state["previous"] = prediction
                    continue
                state = advance(
                    state, velocity, prediction, t_steps[index + 1] - t_steps[index])
                if mode == "memory_zero":
                    state["previous"] = restore_cond(
                        torch.zeros_like(state["previous"]),
                        state["cond_seq"], state["cond_mask"])
                elif mode == "memory_fixed_after_call1":
                    if fixed_memory is None:
                        fixed_memory = prediction.clone()
                    state["previous"] = fixed_memory

            final_latent = last_prediction if mode == "z_frozen_recurrence" else state["z"]
            final_state = dict(state)
            final_state["z"] = final_latent
            token_correct, mask, exact = decode_correctness(
                final_state, model, config, args.self_cond_cfg)
            for local in range(len(exact)):
                final_rows.append({
                    "example_index": seen + local // args.restarts,
                    "restart": local % args.restarts,
                    "mode": mode,
                    "final_exact": bool(exact[local]),
                    "final_target_token_accuracy": float(
                        token_correct[local][mask[local]].float().mean()),
                })
        seen += bsz
        print(f"Processed {seen}/{len(dataset)}", flush=True)

    summary = aggregate(rows)
    final_summary = []
    for mode in modes:
        selected = [x for x in final_rows if x["mode"] == mode]
        final_summary.append({
            "mode": mode,
            "trajectories": len(selected),
            "final_exact_accuracy": float(np.mean([x["final_exact"] for x in selected])),
            "final_target_token_accuracy": float(np.mean([
                x["final_target_token_accuracy"] for x in selected])),
        })
    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": checkpoint_step,
        "difficulty": args.difficulty,
        "samples": seen,
        "restarts": args.restarts,
        "flow_steps": args.flow_steps,
        "summary": summary,
        "final_summary": final_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    write_csv(output_dir / "call_curve.csv", summary)
    write_csv(output_dir / "per_trajectory_calls.csv", rows)
    write_csv(output_dir / "final_summary.csv", final_summary)
    write_csv(output_dir / "per_trajectory_final.csv", final_rows)
    make_plot(summary, output_dir)
    print(json.dumps(final_summary, indent=2), flush=True)
    print(f"Saved compute decomposition to {output_dir}", flush=True)


if __name__ == "__main__":
    run(parse_args())
