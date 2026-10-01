"""Build verifier-backed Countdown or 4x4 Sudoku datasets for official ELF."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random

from datasets import Dataset
from transformers import AutoTokenizer

from countdown_task import (episode as countdown_episode,
                            structured_episode as structured_countdown_episode,
                            verify as verify_countdown)
from sudoku9_task import episode as sudoku_episode, verify as verify_sudoku


def countdown_row(rng, level):
    row = (countdown_episode(rng, level) if level <= 4 else
           structured_countdown_episode(rng, level, rng.choice(("left", "right"))))
    assert verify_countdown(row)
    numbers = " ".join(map(str, row["numbers"]))
    row["input"] = (
        f"Countdown numbers: {numbers}. Target: {row['target']}. "
        "Use every number exactly once with + - * / and give a valid RPN expression:"
    )
    row["target_text"] = row["answer"]
    row["task"] = "countdown"
    row["group"] = f"countdown/d{level}"
    row["key"] = f"{tuple(row['numbers'])}:{row['target']}"
    return row


def sudoku_row(rng, level):
    names = {0: "easy", 1: "medium", 2: "hard"}
    if level not in names:
        raise ValueError("Sudoku levels are 0=easy, 1=medium, 2=hard")
    for _ in range(500):
        blanks = rng.randint(*( (25, 40) if level == 0 else (42, 55) if level == 1 else (48, 58) ))
        row = sudoku_episode(rng, blanks)
        if level == 0 and row["max_guess_depth"] == 0 and row["propagation_rounds"] <= 6:
            break
        if level == 1 and row["max_guess_depth"] <= 1 and row["propagation_rounds"] >= 7:
            break
        if level == 2 and row["max_guess_depth"] >= 2:
            break
    else:
        raise RuntimeError(f"could not sample Sudoku difficulty level {level}")
    assert verify_sudoku(row)
    puzzle = " ".join("." if value == 0 else str(value) for value in row["puzzle"])
    row["input"] = (
        f"Solve this 9 by 9 Sudoku in row-major order: {puzzle}. "
        "Return the 81 entries in row-major order:"
    )
    row["target_text"] = " ".join(map(str, row["answer"]))
    row["difficulty"] = names[level]
    row["task"] = "sudoku"
    row["depth"] = level
    row["group"] = f"sudoku/{names[level]}"
    row["key"] = "".join(map(str, row["puzzle"]))
    return row


def generate_split(task, count, seed, seen, levels, allow_repeats=False, forbidden=None):
    rng = random.Random(seed)
    maker = countdown_row if task == "countdown" else sudoku_row
    rows, attempts = [], 0
    while len(rows) < count:
        level = levels[len(rows) % len(levels)]
        row = maker(rng, level)
        attempts += 1
        if forbidden is not None and row["key"] in forbidden:
            continue
        if row["key"] in seen and not allow_repeats:
            if attempts > count * 100:
                raise RuntimeError("too many duplicate generated problems")
            continue
        if row["key"] not in seen:
            seen.add(row["key"])
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("countdown", "sudoku"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--train-examples", type=int, default=100_000)
    parser.add_argument("--eval-examples", type=int, default=1_200)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--train-levels", default=None,
                        help="comma-separated depths (Countdown) or blank counts (Sudoku)")
    parser.add_argument("--eval-levels", default=None)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    defaults = ((1, 2, 3, 4), (1, 2, 3, 4, 5, 6, 7)) if args.task == "countdown" else ((0, 1, 2), (0, 1, 2))
    train_levels = tuple(map(int, args.train_levels.split(","))) if args.train_levels else defaults[0]
    eval_levels = tuple(map(int, args.eval_levels.split(","))) if args.eval_levels else defaults[1]
    args.out.mkdir(parents=True)
    seen, manifest = set(), {"task": args.task, "seed": args.seed, "splits": {}}
    # Reserve held-out keys before producing training data. Training may repeat
    # within its split, but can never contain a validation/test problem.
    generated = {}
    for offset, split in enumerate(("validation", "test"), start=1):
        levels = eval_levels
        generated[split] = generate_split(
            args.task, args.eval_examples, args.seed + offset, seen, levels,
            allow_repeats=False,
        )
    heldout = set(seen)
    generated["train"] = generate_split(
        args.task, args.train_examples, args.seed, set(), train_levels,
        allow_repeats=True, forbidden=heldout,
    )
    for split in ("train", "validation", "test"):
        count = args.train_examples if split == "train" else args.eval_examples
        rows = generated[split]
        converted = []
        for index, row in enumerate(rows):
            item = dict(row)
            item.update({
                "condition_input_ids": tokenizer(row["input"], add_special_tokens=False)["input_ids"],
                "input_ids": tokenizer(row["target_text"], add_special_tokens=False)["input_ids"],
                "target": row["target_text"], "example_id": f"{split}-{index}",
            })
            converted.append(item)
        Dataset.from_list(converted).save_to_disk(str(args.out / split))
        manifest["splits"][split] = {
            "examples": len(converted),
            "groups": dict(sorted(Counter(row["group"] for row in rows).items())),
            "condition_tokens_max": max(len(row["condition_input_ids"]) for row in converted),
            "target_tokens_max": max(len(row["input_ids"]) for row in converted),
        }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
