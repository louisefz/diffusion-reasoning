#!/usr/bin/env python
"""Test whether an ELF Sudoku flow causally repairs injected wrong digits.

Only trajectories that solve the held-out puzzle before intervention are used.
At selected flow times, one or more blank-cell representations are replaced by
the encoder representation of a deliberately wrong completed grid.  The edit
is applied to the transported state z, the self-conditioning prediction, or
both.  We then finish the untouched native ODE and measure recovery.
"""

import argparse
import csv
import json
import random
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
from official_reasoning_velocity_transport import advance, finish, forward_velocity
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split
from utils.encoder_utils import encode_text
from utils.generation_utils import _dlm_decode_batch
from utils.sampling_utils import get_sampling_steps


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--eval-data-path", default=None)
    p.add_argument("--difficulty", default="hard")
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--flow-steps", type=int, default=16)
    p.add_argument("--time-index", action="append", type=int, default=None)
    p.add_argument("--error-count", action="append", type=int, default=None)
    p.add_argument("--mode", action="append", choices=("z", "previous", "both"), default=None)
    p.add_argument("--time-schedule", choices=("uniform", "logit_normal"), default="uniform")
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--self-cond-cfg", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20261001)
    return p.parse_args()


def subset_state(state, indices):
    batch = state["z"].shape[0]
    result = {}
    for key, value in state.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch:
            result[key] = value.index_select(0, indices)
        else:
            result[key] = value
    return result


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["mode"], row["error_count"], row["time_index"])].append(row)
    result = []
    for (mode, count, time_index), items in sorted(buckets.items()):
        effective = [x for x in items if x["immediate_wrong_digit_rate"] >= 1.0 - 1e-6]
        result.append({
            "mode": mode,
            "error_count": count,
            "time_index": time_index,
            "flow_time": items[0]["flow_time"],
            "eligible_trajectories": len(items),
            "immediate_wrong_digit_rate": float(np.mean([
                x["immediate_wrong_digit_rate"] for x in items
            ])),
            "fully_effective_interventions": len(effective),
            "fully_effective_rate": len(effective) / len(items),
            "exact_recovery_given_fully_effective": float(np.mean([
                x["final_exact_recovered"] for x in effective
            ])) if effective else None,
            "cell_recovery_given_fully_effective": float(np.mean([
                x["injected_cell_recovery_rate"] for x in effective
            ])) if effective else None,
            "final_exact_recovery_rate": float(np.mean([
                x["final_exact_recovered"] for x in items
            ])),
            "injected_cell_recovery_rate": float(np.mean([
                x["injected_cell_recovery_rate"] for x in items
            ])),
            "final_target_token_accuracy": float(np.mean([
                x["final_target_token_accuracy"] for x in items
            ])),
        })
    return result


def make_plot(summary, output_dir, modes, counts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(modes), figsize=(5.2 * len(modes), 7.5),
                             sharex="col", squeeze=False)
    for column, mode in enumerate(modes):
        chosen = [x for x in summary if x["mode"] == mode]
        for count in counts:
            curve = sorted([x for x in chosen if x["error_count"] == count],
                           key=lambda x: x["flow_time"])
            axes[0, column].plot(
                [x["flow_time"] for x in curve],
                [x["final_exact_recovery_rate"] for x in curve],
                marker="o", label=f"{count} wrong cells")
            axes[1, column].plot(
                [x["flow_time"] for x in curve],
                [x["injected_cell_recovery_rate"] for x in curve],
                marker="o", label=f"{count} wrong cells")
        axes[0, column].set_title(f"Inject into {mode}")
        axes[0, column].set_ylabel("Whole-grid recovery")
        axes[1, column].set_ylabel("Injected-cell recovery")
        axes[1, column].set_xlabel("Flow time of intervention")
        for axis in axes[:, column]:
            axis.set_ylim(-.03, 1.03)
            axis.grid(alpha=.25)
            axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Can the native ELF flow repair controlled Sudoku errors?")
    fig.tight_layout()
    fig.savefig(output_dir / "sudoku_error_recovery.png", dpi=220)
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
    if args.difficulty is not None:
        dataset = dataset.filter(lambda row: str(row["difficulty"]) == args.difficulty)
    dataset = dataset.select(range(min(args.num_samples, len(dataset))))
    loader = get_dataloader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        drop_last=False, max_seq_length=config.max_length,
        pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
        max_input_seq_length=config.max_input_length, distributed=False,
    )

    digit_ids = {}
    for digit in range(1, 10):
        ids = tokenizer.encode(str(digit), add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Digit {digit} does not map to exactly one token: {ids}")
        digit_ids[digit] = ids[0]
    next_digit_id = {
        digit_ids[digit]: digit_ids[(digit % 9) + 1] for digit in range(1, 10)
    }

    dtype = next(model.parameters()).dtype
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    t_steps = get_sampling_steps(
        args.flow_steps, args.time_schedule, config.denoiser_p_mean,
        config.denoiser_p_std, device=device, dtype=dtype,
    )
    time_indices = args.time_index or [1, 2, 4, 8, 12]
    counts = sorted(set(args.error_count or [1, 3, 5]))
    modes = args.mode or ["z", "previous", "both"]
    if any(index < 0 or index >= args.flow_steps for index in time_indices):
        raise ValueError("time indices must select a native velocity evaluation")

    rows = []
    seen = 0
    baseline_total = baseline_correct = 0
    use_bf16 = bool(config.use_bf16) and device.type == "cuda"
    for batch in loader:
        bsz = len(batch["target"])
        repeated = repeat_batch(batch, args.restarts)
        noise = torch.randn(
            (bsz * args.restarts, config.max_length, model.text_encoder_dim),
            generator=generator, dtype=dtype,
        )
        state = prepare_flow(repeated, noise, model, encoder, tokenizer, config, device)
        if not bool((state["lengths"] == 81).all()):
            raise ValueError(f"Expected 81 target tokens, got {state['lengths'].tolist()}")
        snapshots = []
        for index in range(args.flow_steps):
            snapshots.append(clone_state(state))
            velocity, prediction = forward_velocity(
                model, state, t_steps[index], config, args.cfg, args.self_cond_cfg)
            state = advance(state, velocity, prediction, t_steps[index + 1] - t_steps[index])
        _, _, final_exact = decode_correctness(state, model, config, args.self_cond_cfg)
        eligible_cpu = torch.where(final_exact.bool())[0]
        baseline_total += len(final_exact)
        baseline_correct += len(eligible_cpu)
        if len(eligible_cpu) == 0:
            seen += bsz
            print(f"Processed {seen}/{len(dataset)}; no baseline-correct trajectories", flush=True)
            continue
        eligible = eligible_cpu.to(device)

        # Select a reproducible nested ordering of originally blank cells for each restart.
        puzzles = dataset[seen:seen + bsz]["puzzle"]
        blank_orders = []
        for local, puzzle in enumerate(puzzles):
            blanks = [i for i, value in enumerate(np.asarray(puzzle).tolist()) if int(value) == 0]
            for restart in range(args.restarts):
                order = list(blanks)
                random.Random(args.seed + (seen + local) * 1009 + restart).shuffle(order)
                blank_orders.append(order)
        chosen_orders = [blank_orders[int(i)] for i in eligible_cpu]

        correct_ids = snapshots[0]["input_ids"].index_select(0, eligible).clone()
        starts = snapshots[0]["starts"].index_select(0, eligible)
        encoder_mask = torch.from_numpy(
            np.asarray(repeated["encoder_attention_mask"])[eligible_cpu.numpy()]
        ).to(device).float()

        for count in counts:
            if any(len(order) < count for order in chosen_orders):
                raise ValueError(f"Not enough blank cells for error-count={count}")
            cell_indices = torch.tensor(
                [order[:count] for order in chosen_orders], device=device, dtype=torch.long)
            absolute_positions = starts[:, None] + cell_indices
            wrong_ids = correct_ids.clone()
            wrong_token_ids = torch.empty_like(absolute_positions)
            for row_index in range(len(eligible_cpu)):
                for column in range(count):
                    position = int(absolute_positions[row_index, column])
                    correct_token = int(correct_ids[row_index, position])
                    if correct_token not in next_digit_id:
                        raise ValueError(f"Unexpected Sudoku token id {correct_token}")
                    wrong_token = next_digit_id[correct_token]
                    wrong_ids[row_index, position] = wrong_token
                    wrong_token_ids[row_index, column] = wrong_token
            wrong_latent = encode_text(
                wrong_ids, encoder_mask, encoder, config.latent_mean,
                config.latent_std, use_bf16=use_bf16,
            ).to(dtype)

            for time_index in time_indices:
                base = subset_state(snapshots[time_index], eligible)
                for mode in modes:
                    branch = clone_state(base)
                    batch_rows = torch.arange(len(eligible), device=device)[:, None].expand_as(
                        absolute_positions)
                    replacement = wrong_latent[batch_rows, absolute_positions]
                    if mode in ("z", "both"):
                        branch["z"][batch_rows, absolute_positions] = replacement
                    if mode in ("previous", "both"):
                        branch["previous"][batch_rows, absolute_positions] = replacement

                    carrier = branch["previous"] if mode == "previous" else branch["z"]
                    immediate_ids = _dlm_decode_batch(
                        carrier, model, 1.0, config, args.self_cond_cfg)
                    immediate_wrong = immediate_ids[batch_rows, absolute_positions].eq(wrong_token_ids)

                    final_state = finish(
                        model, branch, time_index, t_steps, config,
                        args.cfg, args.self_cond_cfg)
                    final_ids = _dlm_decode_batch(
                        final_state["z"], model, 1.0, config, args.self_cond_cfg)
                    positions = torch.arange(final_ids.shape[1], device=device)[None, :]
                    target_mask = ((positions >= branch["starts"][:, None])
                                   & (positions < (branch["starts"] + branch["lengths"])[:, None]))
                    token_correct = final_ids.eq(branch["input_ids"])
                    exact = (token_correct | ~target_mask).all(dim=1)
                    cell_recovered = token_correct[batch_rows, absolute_positions]
                    for local in range(len(eligible_cpu)):
                        rows.append({
                            "example_index": seen + int(eligible_cpu[local]) // args.restarts,
                            "restart": int(eligible_cpu[local]) % args.restarts,
                            "time_index": time_index,
                            "flow_time": float(t_steps[time_index]),
                            "mode": mode,
                            "error_count": count,
                            "immediate_wrong_digit_rate": float(immediate_wrong[local].float().mean()),
                            "final_exact_recovered": bool(exact[local]),
                            "injected_cell_recovery_rate": float(cell_recovered[local].float().mean()),
                            "final_target_token_accuracy": float(
                                token_correct[local][target_mask[local]].float().mean()),
                        })
        seen += bsz
        print(
            f"Processed {seen}/{len(dataset)}; baseline-correct "
            f"{baseline_correct}/{baseline_total}; intervention rows={len(rows)}",
            flush=True,
        )

    summary_rows = aggregate(rows)
    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": checkpoint_step,
        "difficulty": args.difficulty,
        "samples": seen,
        "restarts": args.restarts,
        "baseline_trajectories": baseline_total,
        "baseline_correct_trajectories": baseline_correct,
        "baseline_exact_accuracy": baseline_correct / max(baseline_total, 1),
        "flow_steps": args.flow_steps,
        "time_schedule": args.time_schedule,
        "time_indices": time_indices,
        "error_counts": counts,
        "modes": modes,
        "summary": summary_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "per_trajectory.csv", rows)
    make_plot(summary_rows, output_dir, modes, counts)
    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved controlled error-recovery experiment to {output_dir}", flush=True)


if __name__ == "__main__":
    run(parse_args())
