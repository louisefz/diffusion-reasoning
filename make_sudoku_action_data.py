#!/usr/bin/env python
"""Convert full-state Sudoku transitions into sparse computation actions.

The source dataset already contains the verifier-backed branch position/value.
This conversion changes only the model target: instead of redrawing an almost
unchanged board, the model predicts the computational event that creates the
next state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer

from make_sudoku_transition_data import board_text


def action_text(position: int, value: int, size: int) -> str:
    row, column = divmod(int(position), int(size))
    return f"row {row + 1} column {column + 1} value {int(value)}"


def condition_text(size: int, puzzle, current) -> str:
    return (
        f"Choose one viable {size} by {size} Sudoku search action. "
        f"Clues: {board_text(puzzle)}. Current state: {board_text(current)}. "
        "Choose a minimum-candidate empty cell and a value that permits a full "
        "solution. Return only: row R column C value V."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--num-proc", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    args.output.mkdir(parents=True)
    manifest = {
        "source": str(args.source),
        "representation": "sparse Sudoku action: row R column C value V",
        "splits": {},
    }

    for split in ("train", "validation", "test"):
        dataset = load_from_disk(str(args.source / split))

        def convert(batch):
            prompts, targets, rows, columns = [], [], [], []
            for size, puzzle, current, position, value in zip(
                batch["size"], batch["puzzle"], batch["current"],
                batch["branch_position"], batch["branch_value"],
            ):
                row, column = divmod(int(position), int(size))
                prompts.append(condition_text(int(size), puzzle, current))
                targets.append(action_text(position, value, size))
                rows.append(row + 1)
                columns.append(column + 1)
            return {
                "input": prompts,
                "target": targets,
                "target_text": targets,
                "action_row": rows,
                "action_column": columns,
                "condition_input_ids": tokenizer(
                    prompts, add_special_tokens=False
                )["input_ids"],
                "input_ids": tokenizer(
                    targets, add_special_tokens=False
                )["input_ids"],
            }

        converted = dataset.map(
            convert, batched=True, batch_size=1024, num_proc=args.num_proc,
            desc=f"Converting {split} to sparse actions",
        )
        converted.save_to_disk(str(args.output / split), max_shard_size="512MB")
        target_lengths = [len(ids) for ids in converted["input_ids"]]
        condition_lengths = [len(ids) for ids in converted["condition_input_ids"]]
        manifest["splits"][split] = {
            "rows": len(converted),
            "max_condition_tokens": max(condition_lengths),
            "max_target_tokens": max(target_lengths),
            "min_target_tokens": min(target_lengths),
        }

    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
