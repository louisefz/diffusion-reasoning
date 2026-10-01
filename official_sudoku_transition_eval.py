#!/usr/bin/env python
"""Verifier-backed one-step and autonomous-rollout evaluation for Sudoku flows."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "official-elf" / "src"))

from configs.config import SamplingConfig, load_config_from_yaml
from make_sudoku_transition_data import (
    board_text,
    candidate_values,
    propagate,
    randomized_completion,
)
from modules.t5_encoder import get_encoder
from official_diagnostics import load_model
from utils.data_utils import get_dataloader, get_pad_token_id
from utils.encoder_utils import encode_text
from utils.generation_utils import _dlm_decode_batch, _generate_samples_single_batch
from utils.sampling_utils import get_sampling_steps


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-one-step", type=int, default=256)
    p.add_argument("--one-step-restarts", type=int, default=4)
    p.add_argument("--num-rollout-puzzles", type=int, default=128)
    p.add_argument("--rollout-restarts", type=int, default=4)
    p.add_argument("--max-reasoning-steps", type=int, default=8)
    p.add_argument("--flow-steps", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=20260930)
    return p.parse_args()


def condition_text(size, puzzle, current):
    return (
        f"Advance one valid {size} by {size} Sudoku search step. "
        f"Clues: {board_text(puzzle)}. Current state: {board_text(current)}. "
        "Assign one viable minimum-candidate cell, propagate forced singles, "
        "and return the next state in row-major order:"
    )


def parse_board(text, size):
    # Decode only the generated suffix, then read the first board-sized sequence
    # of board symbols. This tolerates different token lengths for '.' and digits.
    symbols = re.findall(r"(?<!\d)[1-9](?!\d)|\.", text)
    if len(symbols) < size * size:
        return None
    values = tuple(0 if token == "." else int(token) for token in symbols[: size * size])
    if any(value < 0 or value > size for value in values):
        return None
    return values


def valid_partial(grid, size, box):
    if grid is None or len(grid) != size * size:
        return False
    units = []
    units.extend(grid[row * size : (row + 1) * size] for row in range(size))
    units.extend(tuple(grid[row * size + col] for row in range(size)) for col in range(size))
    for br in range(0, size, box):
        for bc in range(0, size, box):
            units.append(tuple(
                grid[(br + dr) * size + bc + dc]
                for dr in range(box) for dc in range(box)
            ))
    return all(len([x for x in unit if x]) == len(set(x for x in unit if x)) for unit in units)


def extends(base, candidate):
    return candidate is not None and all(old == 0 or old == new for old, new in zip(base, candidate))


def is_complete_valid(grid, puzzle, size, box):
    return (
        grid is not None
        and all(grid)
        and extends(puzzle, grid)
        and valid_partial(grid, size, box)
    )


def legal_successors(current, puzzle, size, box, seed):
    """Enumerate all globally viable generator-consistent next states."""
    current = tuple(current)
    empty = [position for position, value in enumerate(current) if not value]
    if not empty:
        return set()
    position = min(empty, key=lambda p: (len(candidate_values(current, p, size, box)), p))
    options = candidate_values(current, position, size, box)
    successors = set()
    for value in options:
        candidate = list(current)
        candidate[position] = value
        candidate = propagate(candidate, size, box)
        if candidate is None:
            continue
        candidate = tuple(candidate)
        if not extends(puzzle, candidate) or not valid_partial(candidate, size, box):
            continue
        # The training generator only keeps branches belonging to a full completion.
        rng = random.Random(seed + position * 101 + value * 1009)
        if randomized_completion(candidate, size, box, rng) is not None:
            successors.add(candidate)
    return successors


def make_generation_rows(items, tokenizer, size):
    # All-dot output reserves the maximum token budget needed by a partial board.
    placeholder = board_text([0] * (size * size))
    output_ids = tokenizer(placeholder, add_special_tokens=False)["input_ids"]
    rows = []
    for item in items:
        text = condition_text(size, item["puzzle"], item["current"])
        rows.append({
            "condition_input_ids": tokenizer(text, add_special_tokens=False)["input_ids"],
            "input_ids": output_ids,
            "input": text,
            "target": placeholder,
        })
    return Dataset.from_list(rows), len(output_ids)


@torch.no_grad()
def generate_boards(items, model, encoder, tokenizer, config, device, size,
                    flow_steps, batch_size, generator):
    dataset, output_budget = make_generation_rows(items, tokenizer, size)
    loader = get_dataloader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        drop_last=False, max_seq_length=config.max_length,
        pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
        max_input_seq_length=config.max_input_length, distributed=False,
    )
    sampling = SamplingConfig(
        sampling_method="ode", num_sampling_steps=[flow_steps], cfgs=[1.0],
        self_cond_cfg_scales=[1.0], time_schedule="logit_normal", sde_gamma=0.0,
    )
    dtype = next(model.parameters()).dtype
    results = []
    for batch in loader:
        input_ids = torch.from_numpy(np.asarray(batch["input_ids"])).to(device).long()
        encoder_mask = torch.from_numpy(np.asarray(batch["encoder_attention_mask"])).to(device).float()
        cond_mask = torch.from_numpy(np.asarray(batch["cond_seq_mask"])).to(device).float()
        starts = cond_mask.long().sum(dim=1)
        cond_seq = encode_text(
            input_ids, encoder_mask, encoder, config.latent_mean, config.latent_std,
            use_bf16=bool(config.use_bf16),
        ).to(dtype)
        z = (torch.randn(
            (len(input_ids), config.max_length, model.text_encoder_dim),
            generator=generator, dtype=dtype,
        ) * config.denoiser_noise_scale).to(device)
        t_steps = get_sampling_steps(
            flow_steps, "logit_normal", config.denoiser_p_mean,
            config.denoiser_p_std, device=device, dtype=dtype,
        )
        latent = _generate_samples_single_batch(
            model=model, generator=generator, z=z, t_steps=t_steps,
            cond_seq=cond_seq, cond_seq_mask=cond_mask, config=config,
            sampling_config=sampling, cfg_scale=1.0, self_cond_cfg_scale=1.0,
        )
        predicted = _dlm_decode_batch(latent, model, 1.0, config, 1.0)
        for row, start in enumerate(starts.tolist()):
            text = tokenizer.decode(
                predicted[row, start : start + output_budget].detach().cpu().tolist(),
                skip_special_tokens=True,
            ).strip()
            results.append((parse_board(text, size), text))
    return results


def evaluate_one_step(rows, args, model, encoder, tokenizer, config, device, size, box, generator):
    selected = [rows[index] for index in range(min(args.num_one_step, len(rows)))]
    items = []
    for index, row in enumerate(selected):
        for restart in range(args.one_step_restarts):
            items.append({
                "source_index": index,
                "restart": restart,
                "puzzle": tuple(map(int, row["puzzle"])),
                "current": tuple(map(int, row["current"])),
                "reference": tuple(map(int, row["next_state"])),
            })
    predictions = generate_boards(
        items, model, encoder, tokenizer, config, device, size,
        args.flow_steps, args.batch_size, generator,
    )
    counters = defaultdict(int)
    per_source = defaultdict(list)
    examples = []
    for item, (prediction, text) in zip(items, predictions):
        legal = legal_successors(
            item["current"], item["puzzle"], size, box,
            args.seed + item["source_index"] * 7919,
        )
        parsed = prediction is not None
        partial = bool(parsed and valid_partial(prediction, size, box))
        preserves_clues = bool(parsed and extends(item["puzzle"], prediction))
        preserves_current = bool(parsed and extends(item["current"], prediction))
        progresses = bool(parsed and sum(v == 0 for v in prediction) < sum(v == 0 for v in item["current"]))
        is_legal = prediction in legal if parsed else False
        exact = prediction == item["reference"] if parsed else False
        completable = False
        if parsed and partial and preserves_clues:
            completable = randomized_completion(
                prediction, size, box,
                random.Random(args.seed + item["source_index"] * 3571 + item["restart"]),
            ) is not None
        for name, value in {
            "parsed": parsed, "partial_valid": partial,
            "preserves_clues": preserves_clues,
            "preserves_current": preserves_current, "progress": progresses,
            "completable": completable, "legal_successor": is_legal,
            "reference_exact": exact,
        }.items():
            counters[name] += int(value)
        per_source[item["source_index"]].append(prediction if is_legal else None)
        if len(examples) < 16:
            examples.append({
                **{k: item[k] for k in ("source_index", "restart")},
                "current": list(item["current"]), "reference": list(item["reference"]),
                "prediction": list(prediction) if prediction else None,
                "decoded": text, "legal_successor": is_legal,
                "reference_exact": exact, "legal_successor_count": len(legal),
            })
    total = len(items)
    sources_with_any = sum(any(x is not None for x in values) for values in per_source.values())
    unique_valid = [len(set(x for x in values if x is not None)) for values in per_source.values()]
    return {
        "source_states": len(selected), "restarts": args.one_step_restarts,
        "trajectories": total,
        **{f"{name}_rate": value / max(total, 1) for name, value in counters.items()},
        "any_legal_successor_rate": sources_with_any / max(len(selected), 1),
        "mean_unique_legal_successors": float(np.mean(unique_valid)) if unique_valid else 0.0,
        "examples": examples,
    }


def unique_puzzles(rows, limit, box):
    result, seen = [], set()
    for row in rows:
        key = str(row["puzzle_key"])
        if key in seen:
            continue
        seen.add(key)
        puzzle = tuple(map(int, row["puzzle"]))
        propagated = propagate(puzzle, int(row["size"]), box)
        if propagated is None:
            continue
        current = tuple(propagated)
        result.append((key, puzzle, current))
        if len(result) >= limit:
            break
    return result


def evaluate_rollout(rows, args, model, encoder, tokenizer, config, device, size, box, generator):
    puzzles = unique_puzzles(rows, args.num_rollout_puzzles, box)
    trajectories = []
    for puzzle_index, (key, puzzle, current) in enumerate(puzzles):
        for restart in range(args.rollout_restarts):
            trajectories.append({
                "puzzle_index": puzzle_index, "puzzle_key": key,
                "restart": restart, "puzzle": puzzle, "current": current,
                "alive": True, "solved": is_complete_valid(current, puzzle, size, box),
                "failure_step": None, "steps": [],
            })
    curve = []
    for reasoning_step in range(1, args.max_reasoning_steps + 1):
        active_indices = [
            index for index, trajectory in enumerate(trajectories)
            if trajectory["alive"] and not trajectory["solved"]
        ]
        items = [{
            "puzzle": trajectories[index]["puzzle"],
            "current": trajectories[index]["current"],
        } for index in active_indices]
        predictions = generate_boards(
            items, model, encoder, tokenizer, config, device, size,
            args.flow_steps, args.batch_size, generator,
        ) if items else []
        valid_this_step = newly_solved = 0
        for index, (prediction, text) in zip(active_indices, predictions):
            trajectory = trajectories[index]
            legal = legal_successors(
                trajectory["current"], trajectory["puzzle"], size, box,
                args.seed + trajectory["puzzle_index"] * 7919 + reasoning_step * 101,
            )
            is_legal = prediction in legal if prediction is not None else False
            trajectory["steps"].append({
                "step": reasoning_step,
                "prediction": list(prediction) if prediction else None,
                "decoded": text,
                "legal": is_legal,
            })
            if not is_legal:
                trajectory["alive"] = False
                trajectory["failure_step"] = reasoning_step
                continue
            valid_this_step += 1
            trajectory["current"] = prediction
            trajectory["solved"] = is_complete_valid(
                prediction, trajectory["puzzle"], size, box
            )
            newly_solved += int(trajectory["solved"])
        total = len(trajectories)
        curve.append({
            "reasoning_steps": reasoning_step,
            "solve_rate": sum(t["solved"] for t in trajectories) / max(total, 1),
            "alive_rate": sum(t["alive"] for t in trajectories) / max(total, 1),
            "valid_transition_rate_among_active": valid_this_step / max(len(active_indices), 1),
            "active_before_step": len(active_indices),
            "newly_solved": newly_solved,
        })
        print(json.dumps(curve[-1]), flush=True)
    per_puzzle_success = []
    for puzzle_index in range(len(puzzles)):
        group = [t for t in trajectories if t["puzzle_index"] == puzzle_index]
        per_puzzle_success.append(any(t["solved"] for t in group))
    return {
        "puzzles": len(puzzles), "restarts": args.rollout_restarts,
        "trajectories": len(trajectories), "curve": curve,
        "any_restart_solve_rate": float(np.mean(per_puzzle_success)) if per_puzzle_success else 0.0,
        "examples": trajectories[:16],
    }


def plot_report(report, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = report["rollout"]["curve"]
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    x = [row["reasoning_steps"] for row in curve]
    ax.plot(x, [100 * row["solve_rate"] for row in curve], "o-", lw=2.4, label="Solved")
    ax.plot(x, [100 * row["alive_rate"] for row in curve], "o--", lw=2.0, label="Still valid")
    ax.set(xlabel="Reasoning transitions K", ylabel="Rate (%)",
           title=f"Autonomous {report['size']}x{report['size']} Sudoku transition rollout")
    ax.set_ylim(0, 101); ax.grid(alpha=.25); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(output_dir / "rollout_curve.png", dpi=220); plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config_from_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name or config.encoder_model_name, local_files_only=True
    )
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    model, checkpoint_step = load_model(
        config, args.checkpoint, encoder_config, tokenizer, device
    )
    dataset = load_from_disk(args.data_path)
    rows = [dataset[index] for index in range(len(dataset))]
    size = int(rows[0]["size"]); box = 2 if size == 4 else 3
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    print("Evaluating verifier-backed one-step transitions...", flush=True)
    one_step = evaluate_one_step(
        rows, args, model, encoder, tokenizer, config, device, size, box, generator
    )
    print(json.dumps({k: v for k, v in one_step.items() if k != "examples"}, indent=2), flush=True)
    print("Evaluating autonomous transition rollout...", flush=True)
    rollout = evaluate_rollout(
        rows, args, model, encoder, tokenizer, config, device, size, box, generator
    )
    report = {
        "checkpoint": args.checkpoint, "checkpoint_step": checkpoint_step,
        "size": size, "flow_steps_per_transition": args.flow_steps,
        "one_step": one_step, "rollout": rollout,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_report(report, output_dir)
    print(f"Saved evaluation to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
