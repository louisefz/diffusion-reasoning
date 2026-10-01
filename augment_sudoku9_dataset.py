#!/usr/bin/env python
"""Expand a tokenized 9x9 Sudoku dataset with exact symmetry transforms.

Every transform preserves Sudoku validity and the solver-derived difficulty of
the source puzzle.  Existing validation/test splits are copied unchanged, and
their exact puzzle keys are forbidden in the expanded training split.
"""

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer


def permutation(rng):
    groups = list(range(3)); rng.shuffle(groups)
    order = []
    for group in groups:
        inside = list(range(3)); rng.shuffle(inside)
        order.extend(group * 3 + i for i in inside)
    return order


def transform_grid(values, rng, digit_map, rows, cols, transpose):
    grid = [list(values[r * 9:(r + 1) * 9]) for r in range(9)]
    grid = [[grid[r][c] for c in cols] for r in rows]
    if transpose:
        grid = [list(x) for x in zip(*grid)]
    return [0 if value == 0 else digit_map[value] for row in grid for value in row]


def transformed(source, rng, tokenizer, example_id):
    digits = list(range(1, 10)); rng.shuffle(digits)
    digit_map = {i + 1: digits[i] for i in range(9)}
    rows, cols = permutation(rng), permutation(rng)
    transpose = bool(rng.randrange(2))
    puzzle = transform_grid(source["puzzle"], rng, digit_map, rows, cols, transpose)
    answer = transform_grid(source["answer"], rng, digit_map, rows, cols, transpose)
    puzzle_text = " ".join("." if value == 0 else str(value) for value in puzzle)
    input_text = (
        f"Solve this 9 by 9 Sudoku in row-major order: {puzzle_text}. "
        "Return the 81 entries in row-major order:"
    )
    target = " ".join(map(str, answer))
    row = dict(source)
    row.update({
        "puzzle": puzzle, "answer": answer, "input": input_text,
        "target_text": target, "target": target,
        "key": "".join(map(str, puzzle)), "example_id": example_id,
        "condition_input_ids": tokenizer(input_text, add_special_tokens=False)["input_ids"],
        "input_ids": tokenizer(target, add_special_tokens=False)["input_ids"],
    })
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--train-examples", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=20260928)
    args = p.parse_args()
    if args.output.exists():
        p.error(f"refusing to overwrite {args.output}")

    train = load_from_disk(str(args.source / "train"))
    validation = load_from_disk(str(args.source / "validation"))
    test = load_from_disk(str(args.source / "test"))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    forbidden = set(validation["key"]) | set(test["key"])
    seen = set(forbidden)
    rng = random.Random(args.seed)
    rows = []

    # Preserve all original examples, then draw source examples cyclically so
    # difficulty remains exactly balanced.
    for source in train:
        if source["key"] not in seen:
            row = dict(source); row["example_id"] = f"train-{len(rows)}"
            rows.append(row); seen.add(source["key"])
    cursor = 0; rejected = 0
    while len(rows) < args.train_examples:
        source = train[cursor % len(train)]; cursor += 1
        row = transformed(source, rng, tokenizer, f"train-{len(rows)}")
        if row["key"] in seen:
            rejected += 1
            continue
        seen.add(row["key"]); rows.append(row)
        if len(rows) % 10_000 == 0:
            print(f"Prepared {len(rows)}/{args.train_examples}", flush=True)

    args.output.mkdir(parents=True)
    Dataset.from_list(rows).save_to_disk(str(args.output / "train"), max_shard_size="512MB")
    # Copy byte-for-byte logically via Dataset to avoid carrying filter caches.
    validation.save_to_disk(str(args.output / "validation"))
    test.save_to_disk(str(args.output / "test"))
    manifest = {
        "source": str(args.source), "seed": args.seed,
        "method": "exact Sudoku digit/row/column/transpose symmetries",
        "rejected_duplicate_or_heldout_keys": rejected,
        "splits": {
            "train": {"examples": len(rows), "groups": dict(Counter(x["group"] for x in rows))},
            "validation": {"examples": len(validation), "unchanged": True},
            "test": {"examples": len(test), "unchanged": True},
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
