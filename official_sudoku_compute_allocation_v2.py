#!/usr/bin/env python
"""Fixed-call allocation between fast memory updates and slow ELF transport."""

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
from utils.sampling_utils import get_sampling_steps


SCHEDULES = {
    # Each list gives the number of fixed-z memory calls before one z update.
    "standard_3": [1, 1, 1],
    "standard_9": [1] * 9,
    "uniform_3x3": [3, 3, 3],
    "early_5_3_1": [5, 3, 1],
    "late_1_3_5": [1, 3, 5],
    "balanced_5flow_9call": [3, 2, 2, 1, 1],
    "standard_16": [1] * 16,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--eval-data-path", default=None)
    p.add_argument("--difficulty", default="hard")
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--restarts", type=int, default=1)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--self-cond-cfg", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20261005)
    return p.parse_args()


def predicted_correctness(state, prediction, model, config, self_cond_cfg):
    predicted = dict(state)
    predicted["z"] = prediction
    return decode_correctness(predicted, model, config, self_cond_cfg)


def aggregate_call_rows(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["schedule"], row["call_index"])].append(row)
    result = []
    for (schedule, call_index), items in sorted(buckets.items()):
        result.append({
            "schedule": schedule,
            "call_index": call_index,
            "cumulative_calls": call_index + 1,
            "outer_index": items[0]["outer_index"],
            "inner_index": items[0]["inner_index"],
            "flow_time": items[0]["flow_time"],
            "z_updates_completed": items[0]["z_updates_completed"],
            "trajectories": len(items),
            "predicted_exact_accuracy": float(np.mean([
                x["predicted_exact"] for x in items])),
            "predicted_target_token_accuracy": float(np.mean([
                x["predicted_target_token_accuracy"] for x in items])),
            "wrong_to_correct": int(sum(x["wrong_to_correct"] for x in items)),
            "correct_to_wrong": int(sum(x["correct_to_wrong"] for x in items)),
        })
    return result


def make_plot(final_summary, call_summary, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nine_call = [x for x in final_summary if x["total_calls"] == 9]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))
    labels = [x["schedule"] for x in nine_call]
    values = [x["predicted_exact_accuracy"] for x in nine_call]
    axes[0].bar(range(len(labels)), values, color="tab:blue", alpha=.8)
    axes[0].set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    axes[0].set_ylabel("Exact-grid accuracy")
    axes[0].set_title("Same budget: 9 denoiser calls")
    axes[0].grid(axis="y", alpha=.25)

    for schedule in SCHEDULES:
        if schedule == "standard_3":
            continue
        curve = sorted([x for x in call_summary if x["schedule"] == schedule],
                       key=lambda x: x["call_index"])
        axes[1].plot([x["cumulative_calls"] for x in curve],
                     [x["predicted_exact_accuracy"] for x in curve],
                     marker="o", label=schedule)
    axes[1].set(xlabel="Cumulative calls", ylabel="Exact-grid accuracy",
                title="Solution quality during allocated computation")
    axes[1].grid(alpha=.25)
    axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle("Inference-time allocation: memory updates vs continuous transport")
    fig.tight_layout()
    fig.savefig(output_dir / "compute_allocation.png", dpi=220)
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
    schedule_times = {
        name: get_sampling_steps(
            len(allocation), "uniform", config.denoiser_p_mean,
            config.denoiser_p_std, device=device, dtype=dtype)
        for name, allocation in SCHEDULES.items()
    }

    call_rows = []
    final_rows = []
    seen = 0
    for batch in loader:
        bsz = len(batch["target"])
        repeated = repeat_batch(batch, args.restarts)
        noise = torch.randn(
            (bsz * args.restarts, config.max_length, model.text_encoder_dim),
            generator=generator, dtype=dtype)
        initial = prepare_flow(repeated, noise, model, encoder, tokenizer, config, device)
        for schedule, allocation in SCHEDULES.items():
            t_steps = schedule_times[schedule]
            state = clone_state(initial)
            previous_exact = None
            call_index = 0
            last_prediction = None
            for outer_index, inner_calls in enumerate(allocation):
                for inner_index in range(inner_calls):
                    velocity, prediction = forward_velocity(
                        model, state, t_steps[outer_index], config,
                        args.cfg, args.self_cond_cfg)
                    token_correct, mask, exact = predicted_correctness(
                        state, prediction, model, config, args.self_cond_cfg)
                    if previous_exact is None:
                        wc = torch.zeros_like(exact, dtype=torch.bool)
                        cw = torch.zeros_like(exact, dtype=torch.bool)
                    else:
                        wc = (~previous_exact) & exact
                        cw = previous_exact & (~exact)
                    for local in range(len(exact)):
                        call_rows.append({
                            "example_index": seen + local // args.restarts,
                            "restart": local % args.restarts,
                            "schedule": schedule,
                            "call_index": call_index,
                            "outer_index": outer_index,
                            "inner_index": inner_index,
                            "flow_time": float(t_steps[outer_index]),
                            "z_updates_completed": outer_index,
                            "predicted_exact": bool(exact[local]),
                            "predicted_target_token_accuracy": float(
                                token_correct[local][mask[local]].float().mean()),
                            "wrong_to_correct": bool(wc[local]),
                            "correct_to_wrong": bool(cw[local]),
                        })
                    previous_exact = exact
                    last_prediction = prediction
                    state["previous"] = prediction
                    call_index += 1
                    # Only the final call assigned to an outer stage transports z.
                    if inner_index == inner_calls - 1:
                        state = advance(
                            state, velocity, prediction,
                            t_steps[outer_index + 1] - t_steps[outer_index])

            predicted_state = dict(state)
            predicted_state["z"] = last_prediction
            pred_token, pred_mask, pred_exact = decode_correctness(
                predicted_state, model, config, args.self_cond_cfg)
            z_token, z_mask, z_exact = decode_correctness(
                state, model, config, args.self_cond_cfg)
            for local in range(len(pred_exact)):
                final_rows.append({
                    "example_index": seen + local // args.restarts,
                    "restart": local % args.restarts,
                    "schedule": schedule,
                    "total_calls": sum(allocation),
                    "z_updates": len(allocation),
                    "predicted_exact": bool(pred_exact[local]),
                    "predicted_target_token_accuracy": float(
                        pred_token[local][pred_mask[local]].float().mean()),
                    "integrated_z_exact": bool(z_exact[local]),
                    "integrated_z_target_token_accuracy": float(
                        z_token[local][z_mask[local]].float().mean()),
                })
        seen += bsz
        print(f"Processed {seen}/{len(dataset)}", flush=True)

    call_summary = aggregate_call_rows(call_rows)
    final_summary = []
    for schedule, allocation in SCHEDULES.items():
        items = [x for x in final_rows if x["schedule"] == schedule]
        final_summary.append({
            "schedule": schedule,
            "total_calls": sum(allocation),
            "z_updates": len(allocation),
            "allocation": allocation,
            "trajectories": len(items),
            "predicted_exact_accuracy": float(np.mean([
                x["predicted_exact"] for x in items])),
            "predicted_target_token_accuracy": float(np.mean([
                x["predicted_target_token_accuracy"] for x in items])),
            "integrated_z_exact_accuracy": float(np.mean([
                x["integrated_z_exact"] for x in items])),
            "integrated_z_target_token_accuracy": float(np.mean([
                x["integrated_z_target_token_accuracy"] for x in items])),
        })
    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": checkpoint_step,
        "difficulty": args.difficulty,
        "samples": seen,
        "restarts": args.restarts,
        "schedules": SCHEDULES,
        "final_summary": final_summary,
        "call_summary": call_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    write_csv(output_dir / "final_summary.csv", final_summary)
    write_csv(output_dir / "call_summary.csv", call_summary)
    write_csv(output_dir / "per_trajectory_final.csv", final_rows)
    write_csv(output_dir / "per_trajectory_calls.csv", call_rows)
    make_plot(final_summary, call_summary, output_dir)
    print(json.dumps(final_summary, indent=2), flush=True)
    print(f"Saved compute-allocation experiment to {output_dir}", flush=True)


if __name__ == "__main__":
    run(parse_args())
