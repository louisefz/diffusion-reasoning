#!/usr/bin/env python
"""Build joint 4x4/9x9 branch-and-propagate transition data for ELF.

Each condition contains the immutable clues and a current partial search state.
Each target is one globally viable MRV branch followed by deterministic naked-
single propagation. Repeated conditions with different targets represent the
multimodal transition distribution explicitly.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

from sudoku_task import random_solution as random_solution4
from sudoku9_task import random_solution as random_solution9


def candidate_values(grid, position, size, box):
    if grid[position]:
        return []
    row, col = divmod(position, size)
    used = set(grid[row * size : (row + 1) * size])
    used |= {grid[r * size + col] for r in range(size)}
    box_row, box_col = row // box * box, col // box * box
    used |= {
        grid[(box_row + r) * size + box_col + c]
        for r in range(box)
        for c in range(box)
    }
    return [value for value in range(1, size + 1) if value not in used]


def propagate(grid, size, box):
    """Apply naked singles to a fixed point; return None on contradiction."""
    result = list(grid)
    while True:
        singles = []
        for position, value in enumerate(result):
            if value:
                continue
            options = candidate_values(result, position, size, box)
            if not options:
                return None
            if len(options) == 1:
                singles.append((position, options[0]))
        if not singles:
            return result
        changed = False
        for position, value in singles:
            if result[position]:
                continue
            if value in candidate_values(result, position, size, box):
                result[position] = value
                changed = True
        if not changed:
            return result


def randomized_completion(puzzle, size, box, rng):
    grid = list(puzzle)

    def search():
        best_position, best_options = None, None
        for position, value in enumerate(grid):
            if value:
                continue
            options = candidate_values(grid, position, size, box)
            if not options:
                return False
            if best_options is None or len(options) < len(best_options):
                best_position, best_options = position, options
                if len(options) == 1:
                    break
        if best_position is None:
            return True
        rng.shuffle(best_options)
        for value in best_options:
            grid[best_position] = value
            if search():
                return True
        grid[best_position] = 0
        return False

    return tuple(grid) if search() else None


def completion_samples(puzzle, size, box, rng, count, attempts=80):
    found = set()
    for _ in range(attempts):
        solution = randomized_completion(puzzle, size, box, rng)
        if solution is not None:
            found.add(solution)
        if len(found) >= count:
            break
    return list(found)


def transition_trajectory(puzzle, solution, size, box, max_decisions):
    current = propagate(puzzle, size, box)
    if current is None:
        return []
    transitions = []
    for decision in range(max_decisions):
        empty = [position for position, value in enumerate(current) if not value]
        if not empty:
            break
        position = min(
            empty, key=lambda p: (len(candidate_values(current, p, size, box)), p)
        )
        options = candidate_values(current, position, size, box)
        value = solution[position]
        if len(options) < 2 or value not in options:
            # Forced moves should already have been consumed by propagate.
            break
        next_grid = list(current)
        next_grid[position] = value
        next_grid = propagate(next_grid, size, box)
        if next_grid is None:
            break
        transitions.append((tuple(current), tuple(next_grid), position, value, decision))
        current = next_grid
    return transitions


def board_text(grid):
    return " ".join("." if value == 0 else str(value) for value in grid)


def make_row(size, puzzle, current, target, position, value, decision, tokenizer, row_id, split):
    input_text = (
        f"Advance one valid {size} by {size} Sudoku search step. "
        f"Clues: {board_text(puzzle)}. Current state: {board_text(current)}. "
        "Assign one viable minimum-candidate cell, propagate forced singles, "
        "and return the next state in row-major order:"
    )
    target_text = board_text(target)
    return {
        "size": size,
        "puzzle": list(puzzle),
        "current": list(current),
        "next_state": list(target),
        "branch_position": position,
        "branch_value": value,
        "search_depth": decision,
        "puzzle_key": f"{size}:" + "".join(map(str, puzzle)),
        "transition_key": f"{size}:" + "".join(map(str, current)),
        "split": split,
        "group": f"sudoku-transition/{size}x{size}",
        "example_id": f"{split}-{row_id}",
        "input": input_text,
        "target": target_text,
        "target_text": target_text,
        "condition_input_ids": tokenizer(input_text, add_special_tokens=False)["input_ids"],
        "input_ids": tokenizer(target_text, add_special_tokens=False)["input_ids"],
    }


def rows_for_split(target_rows, split, seed, tokenizer, sizes):
    rng = random.Random(seed)
    rows_by_size = {size: [] for size in sizes}
    base_quota, remainder = divmod(target_rows, len(sizes))
    quotas = {
        size: base_quota + int(index < remainder)
        for index, size in enumerate(sizes)
    }
    size_cursor = 0
    puzzles = {size: 0 for size in sizes}
    max_input_tokens = max_target_tokens = 0
    while any(len(rows_by_size[size]) < quotas[size] for size in sizes):
        size = sizes[size_cursor % len(sizes)]
        size_cursor += 1
        if len(rows_by_size[size]) >= quotas[size]:
            continue
        box = 2 if size == 4 else 3
        solution = random_solution4(rng) if size == 4 else random_solution9(rng)
        clue_count = rng.randint(1, 4) if size == 4 else rng.randint(14, 22)
        clue_positions = rng.sample(range(size * size), clue_count)
        puzzle = tuple(solution[index] if index in clue_positions else 0 for index in range(size * size))
        completions = completion_samples(
            puzzle, size, box, rng, count=8
        )
        if len(completions) < 2:
            continue
        puzzle_rows = []
        for completion in completions:
            for current, target, position, value, decision in transition_trajectory(
                puzzle, completion, size, box, max_decisions=8 if size == 4 else 16
            ):
                row = make_row(
                    size, puzzle, current, target, position, value, decision,
                    tokenizer, len(rows_by_size[size]) + len(puzzle_rows), split,
                )
                max_input_tokens = max(max_input_tokens, len(row["condition_input_ids"]))
                max_target_tokens = max(max_target_tokens, len(row["input_ids"]))
                puzzle_rows.append(row)
        if not puzzle_rows:
            continue
        # Preserve puzzle-level split isolation and approximately 50/50 rows.
        remaining = quotas[size] - len(rows_by_size[size])
        rows_by_size[size].extend(puzzle_rows[:remaining])
        puzzles[size] += 1
        total = sum(map(len, rows_by_size.values()))
        if total and total % 10_000 < len(puzzle_rows):
            print(
                f"{split}: {total}/{target_rows} "
                "(" + ", ".join(
                    f"{size}x{size}={len(rows_by_size[size])}" for size in sizes
                ) + ")",
                flush=True,
            )
    rows = [row for size in sizes for row in rows_by_size[size]]
    rng.shuffle(rows)
    for index, row in enumerate(rows):
        row["example_id"] = f"{split}-{index}"
    return rows, {
        "rows": len(rows),
        "puzzles": {str(key): value for key, value in puzzles.items()},
        "max_condition_tokens": max_input_tokens,
        "max_target_tokens": max_target_tokens,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--train-examples", type=int, default=200_000)
    parser.add_argument("--validation-examples", type=int, default=10_000)
    parser.add_argument("--test-examples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--sizes", type=int, nargs="+", choices=[4, 9], default=[4, 9])
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    manifest = {
        "seed": args.seed,
        "method": "multimodal MRV branch followed by naked-single propagation",
        "sizes": args.sizes,
        "splits": {},
    }
    args.output.mkdir(parents=True)
    for offset, (split, count) in enumerate([
        ("train", args.train_examples),
        ("validation", args.validation_examples),
        ("test", args.test_examples),
    ]):
        rows, stats = rows_for_split(
            count, split, args.seed + offset * 1_000_003, tokenizer, args.sizes
        )
        Dataset.from_list(rows).save_to_disk(
            str(args.output / split), max_shard_size="512MB"
        )
        manifest["splits"][split] = stats
        del rows
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
