#!/usr/bin/env python
"""Generate a strict train-short/test-long 4x4 Sudoku transition dataset."""

import argparse
import json
import random
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

from make_sudoku_transition_data import (
    completion_samples,
    make_row,
    transition_trajectory,
)
from sudoku_task import random_solution


def generate_rows(target_rows, split, seed, tokenizer, min_depth, max_depth):
    rng = random.Random(seed)
    rows = []
    puzzle_count = trajectory_count = 0
    depth_counts = {depth: 0 for depth in range(min_depth, max_depth + 1)}
    while len(rows) < target_rows:
        solution = random_solution(rng)
        clue_count = rng.randint(1, 4)
        clue_positions = rng.sample(range(16), clue_count)
        puzzle = tuple(solution[index] if index in clue_positions else 0 for index in range(16))
        completions = completion_samples(puzzle, 4, 2, rng, count=8)
        accepted = []
        for completion_index, completion in enumerate(completions):
            trajectory = transition_trajectory(puzzle, completion, 4, 2, max_decisions=8)
            depth = len(trajectory)
            if min_depth <= depth <= max_depth:
                accepted.append((completion_index, trajectory))
        if not accepted:
            continue
        puzzle_count += 1
        for completion_index, trajectory in accepted:
            trajectory_id = f"{split}-p{puzzle_count}-c{completion_index}"
            depth = len(trajectory)
            depth_counts[depth] += 1
            trajectory_count += 1
            for current, target, position, value, decision in trajectory:
                row = make_row(
                    4, puzzle, current, target, position, value, decision,
                    tokenizer, len(rows), split,
                )
                row["trajectory_depth"] = depth
                row["trajectory_id"] = trajectory_id
                rows.append(row)
            if len(rows) >= target_rows:
                break
        if len(rows) // 10_000 > (len(rows) - sum(len(x[1]) for x in accepted)) // 10_000:
            print(f"{split}: {len(rows)}/{target_rows}", flush=True)
    rng.shuffle(rows)
    for index, row in enumerate(rows):
        row["example_id"] = f"{split}-{index}"
    return rows, {
        "rows": len(rows), "puzzles": puzzle_count,
        "trajectories": trajectory_count,
        "trajectory_depth_counts": depth_counts,
        "min_trajectory_depth": min_depth,
        "max_trajectory_depth": max_depth,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--train-examples", type=int, default=200_000)
    p.add_argument("--validation-examples", type=int, default=10_000)
    p.add_argument("--test-examples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=20261001)
    args = p.parse_args()
    if args.output.exists():
        p.error(f"refusing to overwrite {args.output}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    args.output.mkdir(parents=True)
    manifest = {
        "task": "4x4 Sudoku strict reasoning-depth extrapolation",
        "train_depths": [1, 2, 3], "evaluation_depths": [4, 5, 6],
        "seed": args.seed, "splits": {},
    }
    for offset, (split, count, lo, hi) in enumerate([
        ("train", args.train_examples, 1, 3),
        ("validation", args.validation_examples, 4, 6),
        ("test", args.test_examples, 4, 6),
    ]):
        rows, stats = generate_rows(
            count, split, args.seed + offset * 1_000_003,
            tokenizer, lo, hi,
        )
        Dataset.from_list(rows).save_to_disk(
            str(args.output / split), max_shard_size="512MB"
        )
        manifest["splits"][split] = stats
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
