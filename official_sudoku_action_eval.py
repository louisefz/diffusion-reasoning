#!/usr/bin/env python
"""Verifier-backed evaluation of sparse Sudoku computation actions."""

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
from make_sudoku_action_data import action_text, condition_text
from make_sudoku_transition_data import candidate_values, propagate, randomized_completion
from modules.t5_encoder import get_encoder
from official_diagnostics import load_model
from official_sudoku_transition_eval import extends, is_complete_valid, valid_partial
from utils.data_utils import get_dataloader, get_pad_token_id
from utils.encoder_utils import encode_text
from utils.generation_utils import _dlm_decode_batch, _generate_samples_single_batch
from utils.sampling_utils import get_sampling_steps


ACTION_RE = re.compile(
    r"row\s*[:=]?\s*(\d+)\D+column\s*[:=]?\s*(\d+)\D+value\s*[:=]?\s*(\d+)",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-one-step", type=int, default=256)
    parser.add_argument("--one-step-restarts", type=int, default=4)
    parser.add_argument("--num-rollout-puzzles", type=int, default=64)
    parser.add_argument("--rollout-restarts", type=int, default=4)
    parser.add_argument("--max-reasoning-steps", type=int, default=32)
    parser.add_argument("--flow-steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260930)
    return parser.parse_args()


def parse_action(text, size):
    match = ACTION_RE.search(text)
    if match is None:
        return None
    row, column, value = map(int, match.groups())
    if not (1 <= row <= size and 1 <= column <= size and 1 <= value <= size):
        return None
    return (row - 1) * size + column - 1, value


def apply_action(current, puzzle, action, size, box, seed, require_completion=True):
    """Apply one generator-consistent MRV action and deterministic propagation."""
    details = {
        "empty_cell": False,
        "mrv_position": False,
        "candidate_value": False,
        "partial_valid": False,
        "preserves_clues": False,
        "globally_viable": False,
        "legal_action": False,
    }
    if action is None:
        return None, details
    position, value = action
    current = tuple(current)
    empty = [index for index, item in enumerate(current) if item == 0]
    if not empty or current[position] != 0:
        return None, details
    details["empty_cell"] = True
    mrv = min(empty, key=lambda p: (len(candidate_values(current, p, size, box)), p))
    if position != mrv:
        return None, details
    details["mrv_position"] = True
    if value not in candidate_values(current, position, size, box):
        return None, details
    details["candidate_value"] = True
    successor = list(current)
    successor[position] = value
    successor = propagate(successor, size, box)
    if successor is None:
        return None, details
    successor = tuple(successor)
    details["partial_valid"] = valid_partial(successor, size, box)
    details["preserves_clues"] = extends(puzzle, successor)
    if not details["partial_valid"] or not details["preserves_clues"]:
        return None, details
    details["globally_viable"] = (
        randomized_completion(successor, size, box, random.Random(seed)) is not None
        if require_completion else True
    )
    details["legal_action"] = details["globally_viable"]
    return (successor if details["legal_action"] else None), details


def make_generation_rows(items, tokenizer, size):
    placeholder = action_text(size * size - 1, size, size)
    output_ids = tokenizer(placeholder, add_special_tokens=False)["input_ids"]
    rows = []
    for item in items:
        prompt = condition_text(size, item["puzzle"], item["current"])
        rows.append({
            "condition_input_ids": tokenizer(prompt, add_special_tokens=False)["input_ids"],
            "input_ids": output_ids,
            "input": prompt,
            "target": placeholder,
        })
    return Dataset.from_list(rows), len(output_ids)


@torch.no_grad()
def generate_actions(items, model, encoder, tokenizer, config, device, size,
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
            decoded = tokenizer.decode(
                predicted[row, start:start + output_budget].detach().cpu().tolist(),
                skip_special_tokens=True,
            ).strip()
            results.append((parse_action(decoded, size), decoded))
    return results


def evaluate_one_step(rows, args, model, encoder, tokenizer, config, device,
                      size, box, generator):
    selected = rows[:min(args.num_one_step, len(rows))]
    items = []
    for source_index, row in enumerate(selected):
        for restart in range(args.one_step_restarts):
            items.append({
                "source_index": source_index,
                "restart": restart,
                "puzzle": tuple(map(int, row["puzzle"])),
                "current": tuple(map(int, row["current"])),
                "reference": tuple(map(int, row["next_state"])),
                "reference_action": (int(row["branch_position"]), int(row["branch_value"])),
            })
    outputs = generate_actions(
        items, model, encoder, tokenizer, config, device, size,
        args.flow_steps, args.batch_size, generator,
    )
    counters = defaultdict(int)
    legal_by_source = defaultdict(list)
    examples = []
    for item, (action, decoded) in zip(items, outputs):
        successor, details = apply_action(
            item["current"], item["puzzle"], action, size, box,
            args.seed + item["source_index"] * 7919 + item["restart"] * 101,
        )
        parsed = action is not None
        exact_action = action == item["reference_action"]
        exact_successor = successor == item["reference"] if successor is not None else False
        counters["parsed"] += int(parsed)
        counters["reference_action_exact"] += int(exact_action)
        counters["reference_successor_exact"] += int(exact_successor)
        for key, value in details.items():
            counters[key] += int(value)
        legal_by_source[item["source_index"]].append(action if details["legal_action"] else None)
        if len(examples) < 20:
            examples.append({
                "source_index": item["source_index"],
                "restart": item["restart"],
                "decoded": decoded,
                "prediction": list(action) if action else None,
                "reference_action": list(item["reference_action"]),
                "legal_action": details["legal_action"],
                "reference_action_exact": exact_action,
            })
    total = len(items)
    unique_legal = [len(set(action for action in values if action is not None))
                    for values in legal_by_source.values()]
    return {
        "source_states": len(selected),
        "restarts": args.one_step_restarts,
        "trajectories": total,
        **{f"{key}_rate": value / max(total, 1) for key, value in counters.items()},
        "any_legal_action_rate": float(np.mean([
            any(action is not None for action in values) for values in legal_by_source.values()
        ])) if legal_by_source else 0.0,
        "mean_unique_legal_actions": float(np.mean(unique_legal)) if unique_legal else 0.0,
        "examples": examples,
    }


def unique_puzzles(rows, limit, size, box):
    result, seen = [], set()
    for row in rows:
        key = str(row["puzzle_key"])
        if key in seen:
            continue
        seen.add(key)
        puzzle = tuple(map(int, row["puzzle"]))
        initial = propagate(puzzle, size, box)
        if initial is not None:
            result.append((key, puzzle, tuple(initial)))
        if len(result) >= limit:
            break
    return result


def evaluate_rollout(rows, args, model, encoder, tokenizer, config, device,
                     size, box, generator):
    puzzles = unique_puzzles(rows, args.num_rollout_puzzles, size, box)
    trajectories = []
    for puzzle_index, (key, puzzle, initial) in enumerate(puzzles):
        for restart in range(args.rollout_restarts):
            trajectories.append({
                "puzzle_index": puzzle_index,
                "puzzle_key": key,
                "restart": restart,
                "puzzle": puzzle,
                "current": initial,
                "alive": True,
                "solved": is_complete_valid(initial, puzzle, size, box),
                "failure_step": None,
                "steps": [],
            })
    curve = []
    for reasoning_step in range(1, args.max_reasoning_steps + 1):
        active = [index for index, item in enumerate(trajectories)
                  if item["alive"] and not item["solved"]]
        requests = [{"puzzle": trajectories[index]["puzzle"],
                     "current": trajectories[index]["current"]} for index in active]
        outputs = generate_actions(
            requests, model, encoder, tokenizer, config, device, size,
            args.flow_steps, args.batch_size, generator,
        ) if requests else []
        valid_this_step = newly_solved = 0
        for index, (action, decoded) in zip(active, outputs):
            trajectory = trajectories[index]
            successor, details = apply_action(
                trajectory["current"], trajectory["puzzle"], action, size, box,
                args.seed + trajectory["puzzle_index"] * 7919
                + trajectory["restart"] * 1009 + reasoning_step * 101,
            )
            trajectory["steps"].append({
                "step": reasoning_step,
                "decoded": decoded,
                "action": list(action) if action else None,
                "legal": details["legal_action"],
            })
            if successor is None:
                trajectory["alive"] = False
                trajectory["failure_step"] = reasoning_step
                continue
            valid_this_step += 1
            trajectory["current"] = successor
            trajectory["solved"] = is_complete_valid(
                successor, trajectory["puzzle"], size, box
            )
            newly_solved += int(trajectory["solved"])
        total = len(trajectories)
        point = {
            "reasoning_steps": reasoning_step,
            "solve_rate": sum(item["solved"] for item in trajectories) / max(total, 1),
            "alive_rate": sum(item["alive"] for item in trajectories) / max(total, 1),
            "valid_action_rate_among_active": valid_this_step / max(len(active), 1),
            "active_before_step": len(active),
            "newly_solved": newly_solved,
        }
        curve.append(point)
        print(json.dumps(point), flush=True)
    per_puzzle = []
    for puzzle_index in range(len(puzzles)):
        group = [item for item in trajectories if item["puzzle_index"] == puzzle_index]
        per_puzzle.append(any(item["solved"] for item in group))
    return {
        "puzzles": len(puzzles),
        "restarts": args.rollout_restarts,
        "trajectories": len(trajectories),
        "curve": curve,
        "any_restart_solve_rate": float(np.mean(per_puzzle)) if per_puzzle else 0.0,
        "examples": trajectories[:20],
    }


def plot_report(report, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = report["rollout"]["curve"]
    fig, axis = plt.subplots(figsize=(6.6, 4.6))
    x = [point["reasoning_steps"] for point in curve]
    axis.plot(x, [100 * point["solve_rate"] for point in curve], "o-", label="Solved")
    axis.plot(x, [100 * point["alive_rate"] for point in curve], "o--", label="Valid trajectory")
    axis.set(xlabel="Explicit reasoning transitions K", ylabel="Rate (%)",
             title=f"{report['size']}x{report['size']} Sudoku action-flow rollout")
    axis.set_ylim(0, 101)
    axis.grid(alpha=.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "action_rollout_curve.png", dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
    size = int(rows[0]["size"])
    box = 2 if size == 4 else 3
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print("Evaluating verifier-backed sparse actions...", flush=True)
    one_step = evaluate_one_step(
        rows, args, model, encoder, tokenizer, config, device, size, box, generator
    )
    print(json.dumps({k: v for k, v in one_step.items() if k != "examples"}, indent=2), flush=True)
    print("Evaluating autonomous action rollout...", flush=True)
    rollout = evaluate_rollout(
        rows, args, model, encoder, tokenizer, config, device, size, box, generator
    )
    report = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": checkpoint_step,
        "size": size,
        "flow_steps_per_transition": args.flow_steps,
        "one_step": one_step,
        "rollout": rollout,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_report(report, output_dir)
    print(f"Saved evaluation to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
