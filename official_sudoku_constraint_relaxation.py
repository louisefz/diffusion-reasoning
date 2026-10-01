#!/usr/bin/env python
"""Trace Sudoku constraint energy after controlled ELF flow perturbations."""

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
from official_reasoning_velocity_transport import advance, forward_velocity
from official_sudoku_error_injection import subset_state, write_csv
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
    p.add_argument("--time-index", type=int, default=12)
    p.add_argument("--error-count", action="append", type=int, default=None)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--self-cond-cfg", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20261002)
    return p.parse_args()


def duplicate_energy(values):
    valid = [value for value in values if 1 <= value <= 9]
    return len(valid) - len(set(valid))


def structural_energy(grid):
    """Row/column/box duplicate energy; no answer or clue information."""
    rows = sum(duplicate_energy(grid[9 * r:9 * r + 9]) for r in range(9))
    columns = sum(
        duplicate_energy([grid[9 * r + c] for r in range(9)]) for c in range(9))
    boxes = 0
    for br in range(3):
        for bc in range(3):
            boxes += duplicate_energy([
                grid[9 * r + c]
                for r in range(3 * br, 3 * br + 3)
                for c in range(3 * bc, 3 * bc + 3)
            ])
    return rows + columns + boxes


def search_wrong_assignment(answer, cells, seed, trials=512):
    """Return minimum- and maximum-conflict assignments on identical cells.

    A zero-conflict wrong full grid cannot exist for a unique Sudoku while all
    clues are held fixed.  Random search therefore supplies a matched
    minimum-conflict control instead of claiming an impossible zero-conflict
    completion.
    """
    rng = random.Random(seed)
    best_low = best_high = None
    low_energy = float("inf")
    high_energy = -1
    for _ in range(trials):
        values = []
        candidate = list(map(int, answer))
        for cell in cells:
            options = [digit for digit in range(1, 10) if digit != int(answer[cell])]
            value = rng.choice(options)
            candidate[cell] = value
            values.append(value)
        energy = structural_energy(candidate)
        if energy < low_energy:
            low_energy, best_low = energy, list(values)
        if energy > high_energy:
            high_energy, best_high = energy, list(values)
    return {
        "low_conflict": (best_low, low_energy),
        "high_conflict": (best_high, high_energy),
    }


def peer_mask(cells):
    selected = set(map(int, cells))
    peers = set()
    for cell in selected:
        row, column = divmod(cell, 9)
        peers.update(9 * row + c for c in range(9))
        peers.update(9 * r + column for r in range(9))
        br, bc = row // 3, column // 3
        peers.update(9 * r + c
                     for r in range(3 * br, 3 * br + 3)
                     for c in range(3 * bc, 3 * bc + 3))
    peers -= selected
    return selected, peers, set(range(81)) - selected - peers


def sudoku_metrics(token_ids, digit_by_id, puzzles, answers):
    metrics = []
    for tokens, puzzle, answer in zip(token_ids, puzzles, answers):
        grid = [digit_by_id.get(int(token), 0) for token in tokens]
        invalid = sum(value == 0 for value in grid)
        row_duplicates = sum(duplicate_energy(grid[9 * r:9 * r + 9]) for r in range(9))
        column_duplicates = sum(
            duplicate_energy([grid[9 * r + c] for r in range(9)]) for c in range(9))
        box_duplicates = 0
        for br in range(3):
            for bc in range(3):
                box = [grid[9 * r + c]
                       for r in range(3 * br, 3 * br + 3)
                       for c in range(3 * bc, 3 * bc + 3)]
                box_duplicates += duplicate_energy(box)
        clue_mismatches = sum(
            int(int(clue) != 0 and grid[i] != int(clue))
            for i, clue in enumerate(puzzle))
        answer_hamming = sum(grid[i] != int(answer[i]) for i in range(81))
        constraint_energy = (
            invalid + row_duplicates + column_duplicates + box_duplicates + clue_mismatches)
        metrics.append({
            "constraint_energy": constraint_energy,
            "invalid_tokens": invalid,
            "row_duplicates": row_duplicates,
            "column_duplicates": column_duplicates,
            "box_duplicates": box_duplicates,
            "clue_mismatches": clue_mismatches,
            "answer_hamming": answer_hamming,
            "exact": answer_hamming == 0,
        })
    return metrics


def decode_target(state, latent, model, config, self_cond_cfg):
    ids = _dlm_decode_batch(latent, model, 1.0, config, self_cond_cfg)
    rows = torch.arange(ids.shape[0], device=ids.device)[:, None]
    offsets = torch.arange(81, device=ids.device)[None, :]
    return ids[rows, state["starts"][:, None] + offsets].cpu().numpy()


def matched_random_replacement(current, semantic, generator):
    delta = semantic.float() - current.float()
    noise = torch.randn(delta.shape, generator=generator, dtype=torch.float32).to(delta.device)
    noise = noise / noise.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    noise = noise * delta.norm(dim=-1, keepdim=True)
    return (current.float() + noise).to(current.dtype)


def aggregate(rows, by_actual=False):
    buckets = defaultdict(list)
    for row in rows:
        count = row["actual_changed_count"] if by_actual else row["error_count"]
        buckets[(row["condition"], count, row["state_index"], row["carrier"])].append(row)
    result = []
    for (condition, count, state_index, carrier), items in sorted(buckets.items()):
        result.append({
            "condition": condition,
            "error_count": count,
            "state_index": state_index,
            "flow_time": items[0]["flow_time"],
            "carrier": carrier,
            "trajectories": len(items),
            "nominal_error_count_mean": float(np.mean([x["error_count"] for x in items])),
            "actual_changed_count_mean": float(np.mean([
                x["actual_changed_count"] for x in items])),
            "latent_delta_norm_mean": float(np.mean([x["latent_delta_norm"] for x in items])),
            "constraint_energy": float(np.mean([x["constraint_energy"] for x in items])),
            "answer_hamming": float(np.mean([x["answer_hamming"] for x in items])),
            "exact_rate": float(np.mean([x["exact"] for x in items])),
            "row_duplicates": float(np.mean([x["row_duplicates"] for x in items])),
            "column_duplicates": float(np.mean([x["column_duplicates"] for x in items])),
            "box_duplicates": float(np.mean([x["box_duplicates"] for x in items])),
            "clue_mismatches": float(np.mean([x["clue_mismatches"] for x in items])),
            "invalid_tokens": float(np.mean([x["invalid_tokens"] for x in items])),
            "peer_difference_rate": float(np.mean([
                x["peer_difference_rate"] for x in items])),
            "nonpeer_difference_rate": float(np.mean([
                x["nonpeer_difference_rate"] for x in items])),
        })
    return result


def make_plot(summary, output_dir, counts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = ["low_conflict_native", "high_conflict_native",
                  "high_conflict_frozen_previous", "random_matched_native"]
    fig, axes = plt.subplots(2, len(conditions), figsize=(5.2 * len(conditions), 7.5),
                             sharex="col", squeeze=False)
    for column, condition in enumerate(conditions):
        selected = [x for x in summary
                    if x["condition"] == condition and x["carrier"] == "z"]
        for count in counts:
            curve = sorted([x for x in selected if x["error_count"] == count],
                           key=lambda x: x["flow_time"])
            axes[0, column].plot([x["flow_time"] for x in curve],
                                 [x["constraint_energy"] for x in curve],
                                 marker="o", label=f"{count} edits")
            axes[1, column].plot([x["flow_time"] for x in curve],
                                 [x["answer_hamming"] for x in curve],
                                 marker="o", label=f"{count} edits")
        axes[0, column].set_title(condition.replace("_", " "))
        axes[0, column].set_ylabel("Sudoku constraint energy")
        axes[1, column].set_ylabel("Distance to correct answer")
        axes[1, column].set_xlabel("Flow time")
        for axis in axes[:, column]:
            axis.grid(alpha=.25)
            axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Constraint relaxation after a late-flow perturbation")
    fig.tight_layout()
    fig.savefig(output_dir / "constraint_relaxation.png", dpi=220)
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

    digit_ids = {}
    for digit in range(1, 10):
        ids = tokenizer.encode(str(digit), add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Digit {digit} tokenization is not atomic: {ids}")
        digit_ids[digit] = ids[0]
    digit_by_id = {token: digit for digit, token in digit_ids.items()}
    next_digit_id = {digit_ids[d]: digit_ids[d % 9 + 1] for d in range(1, 10)}

    dataset = load_dataset_split(args.eval_data_path or config.eval_data_path)
    dataset = dataset.filter(lambda row: str(row["difficulty"]) == args.difficulty)
    dataset = dataset.select(range(min(args.num_samples, len(dataset))))
    loader = get_dataloader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        drop_last=False, max_seq_length=config.max_length,
        pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
        max_input_seq_length=config.max_input_length, distributed=False)

    dtype = next(model.parameters()).dtype
    noise_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    random_generator = torch.Generator(device="cpu").manual_seed(args.seed + 17)
    t_steps = get_sampling_steps(
        args.flow_steps, "uniform", config.denoiser_p_mean, config.denoiser_p_std,
        device=device, dtype=dtype)
    counts = sorted(set(args.error_count or [1, 3, 5, 10, 20]))
    use_bf16 = bool(config.use_bf16) and device.type == "cuda"
    rows = []
    cell_rows = []
    seen = total = eligible_total = 0

    for batch in loader:
        bsz = len(batch["target"])
        repeated = repeat_batch(batch, args.restarts)
        noise = torch.randn(
            (bsz * args.restarts, config.max_length, model.text_encoder_dim),
            generator=noise_generator, dtype=dtype)
        state = prepare_flow(repeated, noise, model, encoder, tokenizer, config, device)
        injection_state = None
        for index in range(args.flow_steps):
            if index == args.time_index:
                injection_state = clone_state(state)
            velocity, prediction = forward_velocity(
                model, state, t_steps[index], config, args.cfg, args.self_cond_cfg)
            state = advance(state, velocity, prediction, t_steps[index + 1] - t_steps[index])
        if injection_state is None:
            raise ValueError("time-index did not select a flow state")
        _, _, exact = decode_correctness(state, model, config, args.self_cond_cfg)
        eligible_cpu = torch.where(exact.bool())[0]
        total += len(exact)
        eligible_total += len(eligible_cpu)
        if len(eligible_cpu) == 0:
            seen += bsz
            continue
        eligible = eligible_cpu.to(device)
        base = subset_state(injection_state, eligible)

        base_rows = dataset[seen:seen + bsz]
        puzzle_flat = []
        answer_flat = []
        blank_orders = []
        for local, (puzzle, answer) in enumerate(zip(base_rows["puzzle"], base_rows["answer"])):
            blanks = [i for i, value in enumerate(puzzle) if int(value) == 0]
            for restart in range(args.restarts):
                order = list(blanks)
                random.Random(args.seed + (seen + local) * 1009 + restart).shuffle(order)
                blank_orders.append(order)
                puzzle_flat.append(np.asarray(puzzle).tolist())
                answer_flat.append(np.asarray(answer).tolist())
        chosen_orders = [blank_orders[int(i)] for i in eligible_cpu]
        puzzles = [puzzle_flat[int(i)] for i in eligible_cpu]
        answers = [answer_flat[int(i)] for i in eligible_cpu]

        # Paired unperturbed trajectory for spatial propagation controls.
        baseline = clone_state(base)
        baseline_grids = {"z": [], "previous": []}
        for state_index in range(args.time_index, args.flow_steps + 1):
            for carrier in ("z", "previous"):
                baseline_grids[carrier].append(decode_target(
                    baseline, baseline[carrier], model, config, args.self_cond_cfg))
            if state_index < args.flow_steps:
                velocity, prediction = forward_velocity(
                    model, baseline, t_steps[state_index], config,
                    args.cfg, args.self_cond_cfg)
                baseline = advance(
                    baseline, velocity, prediction,
                    t_steps[state_index + 1] - t_steps[state_index])

        correct_ids = base["input_ids"].clone()
        encoder_mask = torch.from_numpy(
            np.asarray(repeated["encoder_attention_mask"])[eligible_cpu.numpy()]
        ).to(device).float()
        for count in counts:
            if any(len(order) < count for order in chosen_orders):
                raise ValueError(f"A puzzle has fewer than {count} blank cells")
            cells = torch.tensor([order[:count] for order in chosen_orders],
                                 device=device, dtype=torch.long)
            positions = base["starts"][:, None] + cells
            batch_rows = torch.arange(len(eligible), device=device)[:, None].expand_as(positions)
            assignments = {"low_conflict": [], "high_conflict": []}
            intended_energy = {"low_conflict": [], "high_conflict": []}
            for local, order in enumerate(chosen_orders):
                found = search_wrong_assignment(
                    answers[local], order[:count],
                    args.seed + 1000003 * (seen + local) + count)
                for label, (values, energy) in found.items():
                    assignments[label].append(values)
                    intended_energy[label].append(energy)

            semantic_by_label = {}
            for label in ("low_conflict", "high_conflict"):
                wrong_ids = correct_ids.clone()
                for row_index in range(len(eligible)):
                    for column in range(count):
                        wrong_ids[row_index, int(positions[row_index, column])] = digit_ids[
                            assignments[label][row_index][column]]
                wrong_latent = encode_text(
                    wrong_ids, encoder_mask, encoder, config.latent_mean, config.latent_std,
                    use_bf16=use_bf16).to(dtype)
                semantic_by_label[label] = wrong_latent[batch_rows, positions]

            branches = {}
            for label in ("low_conflict", "high_conflict"):
                semantic = semantic_by_label[label]
                native = clone_state(base)
                native["z"][batch_rows, positions] = semantic
                native["previous"][batch_rows, positions] = semantic
                branches[f"{label}_native"] = (native, label)
                if label == "high_conflict":
                    branches["high_conflict_frozen_previous"] = (clone_state(native), label)

            high_semantic = semantic_by_label["high_conflict"]
            random_branch = clone_state(base)
            random_branch["z"][batch_rows, positions] = matched_random_replacement(
                base["z"][batch_rows, positions], high_semantic, random_generator)
            random_branch["previous"][batch_rows, positions] = matched_random_replacement(
                base["previous"][batch_rows, positions], high_semantic, random_generator)
            branches["random_matched_native"] = (random_branch, "high_conflict")

            for condition, (branch, source_label) in branches.items():
                frozen_previous = branch["previous"].clone()
                delta_z = (branch["z"][batch_rows, positions].float()
                           - base["z"][batch_rows, positions].float())
                delta_previous = (branch["previous"][batch_rows, positions].float()
                                  - base["previous"][batch_rows, positions].float())
                delta_norm = torch.sqrt(
                    delta_z.square().sum(dim=(1, 2))
                    + delta_previous.square().sum(dim=(1, 2))).cpu().numpy()
                z_history = []
                for state_index in range(args.time_index, args.flow_steps + 1):
                    flow_time = float(t_steps[state_index])
                    for carrier in ("z", "previous"):
                        target_ids = decode_target(
                            branch, branch[carrier], model, config, args.self_cond_cfg)
                        if carrier == "z":
                            z_history.append(np.asarray([
                                [digit_by_id.get(int(token), 0) for token in token_row]
                                for token_row in target_ids
                            ], dtype=np.int16))
                        decoded = sudoku_metrics(target_ids, digit_by_id, puzzles, answers)
                        for local, metric in enumerate(decoded):
                            initial_grid = z_history[0][local]
                            actual_changed = int(sum(
                                int(initial_grid[int(cell)]) != int(answers[local][int(cell)])
                                for cell in cells[local].cpu().tolist()))
                            selected, peers, nonpeers = peer_mask(cells[local].cpu().tolist())
                            baseline_grid = baseline_grids[carrier][
                                state_index - args.time_index][local]
                            peer_difference = sum(
                                int(target_ids[local][cell]) != int(baseline_grid[cell])
                                for cell in peers)
                            nonpeer_difference = sum(
                                int(target_ids[local][cell]) != int(baseline_grid[cell])
                                for cell in nonpeers)
                            rows.append({
                                "example_index": seen + int(eligible_cpu[local]) // args.restarts,
                                "restart": int(eligible_cpu[local]) % args.restarts,
                                "condition": condition,
                                "error_count": count,
                                "actual_changed_count": actual_changed,
                                "intended_structural_energy": (
                                    None if condition == "random_matched_native"
                                    else intended_energy[source_label][local]),
                                "latent_delta_norm": float(delta_norm[local]),
                                "state_index": state_index,
                                "flow_time": flow_time,
                                "carrier": carrier,
                                "peer_difference_from_baseline": peer_difference,
                                "nonpeer_difference_from_baseline": nonpeer_difference,
                                "peer_difference_rate": peer_difference / max(len(peers), 1),
                                "nonpeer_difference_rate": nonpeer_difference / max(len(nonpeers), 1),
                                **metric,
                            })
                    if state_index == args.flow_steps:
                        break
                    velocity, prediction = forward_velocity(
                        model, branch, t_steps[state_index], config,
                        args.cfg, args.self_cond_cfg)
                    branch = advance(
                        branch, velocity, prediction,
                        t_steps[state_index + 1] - t_steps[state_index])
                    if condition == "high_conflict_frozen_previous":
                        branch["previous"] = frozen_previous
                # Cell-level correction times and reversals, kept per trajectory.
                for local in range(len(eligible)):
                    selected_cells = cells[local].cpu().tolist()
                    for cell in selected_cells:
                        sequence = [int(grid[local][cell]) == int(answers[local][cell])
                                    for grid in z_history]
                        initially_wrong = not sequence[0]
                        recovery_offset = next(
                            (offset for offset, correct in enumerate(sequence) if correct), None)
                        reversal = False
                        if recovery_offset is not None:
                            reversal = any(not correct for correct in sequence[recovery_offset + 1:])
                        cell_rows.append({
                            "example_index": seen + int(eligible_cpu[local]) // args.restarts,
                            "restart": int(eligible_cpu[local]) % args.restarts,
                            "condition": condition,
                            "error_count": count,
                            "cell": int(cell),
                            "initially_wrong": initially_wrong,
                            "first_recovery_state": (
                                args.time_index + recovery_offset
                                if initially_wrong and recovery_offset is not None else None),
                            "first_recovery_time": (
                                float(t_steps[args.time_index + recovery_offset])
                                if initially_wrong and recovery_offset is not None else None),
                            "reversed_after_recovery": reversal if initially_wrong else False,
                        })
        seen += bsz
        print(f"Processed {seen}/{len(dataset)}; eligible={eligible_total}/{total}", flush=True)

    summary = aggregate(rows)
    summary_by_actual = aggregate(rows, by_actual=True)
    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": checkpoint_step,
        "difficulty": args.difficulty,
        "samples": seen,
        "restarts": args.restarts,
        "eligible_trajectories": eligible_total,
        "total_trajectories": total,
        "time_index": args.time_index,
        "injection_time": float(t_steps[args.time_index]),
        "error_counts": counts,
        "summary": summary,
        "summary_by_actual_changed_count": summary_by_actual,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "summary_by_actual_changed_count.csv", summary_by_actual)
    write_csv(output_dir / "per_trajectory.csv", rows)
    write_csv(output_dir / "per_cell.csv", cell_rows)
    make_plot(summary, output_dir, counts)
    print(f"Saved constraint-relaxation experiment to {output_dir}", flush=True)


if __name__ == "__main__":
    run(parse_args())
