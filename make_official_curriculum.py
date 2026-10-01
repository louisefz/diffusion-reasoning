"""Build a d4-heavy second-stage curriculum for the official ELF model."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

from make_official_composition import convert
from task import generate, validate


DEFAULT_SPEC = (
    ("compose", 4, 72_000),
    ("compose", 2, 32_000),
    ("compose", 1, 8_000),
    ("lookup", 1, 4_000),
    ("lookup", 2, 4_000),
    ("lookup", 4, 4_000),
    ("lookup", 8, 4_000),
)


def digest(rows):
    hasher = hashlib.sha256()
    for row in rows:
        hasher.update((json.dumps(row, sort_keys=True) + "\n").encode())
    return hasher.hexdigest()


def load_forbidden(base_data):
    forbidden = set()
    split_counts = {}
    for split in ("train", "validation", "test"):
        dataset = load_from_disk(str(base_data / split))
        before = len(forbidden)
        forbidden.update(dataset["table_id"])
        split_counts[split] = len(forbidden) - before
    return forbidden, split_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-data", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--symbols", type=int, default=8)
    parser.add_argument("--functions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument(
        "--scale", type=int, default=1,
        help="Multiply every entry in the default curriculum specification.",
    )
    args = parser.parse_args()
    if args.scale < 1:
        parser.error("--scale must be at least 1")
    if args.out.exists():
        parser.error(f"Refusing to overwrite {args.out}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True,
    )
    forbidden, excluded_split_counts = load_forbidden(args.base_data)
    excluded_count = len(forbidden)
    base_forbidden = set(forbidden)
    source_rows = []
    segment_manifest = []
    for segment, (task_name, depth, base_count) in enumerate(DEFAULT_SPEC):
        count = base_count * args.scale
        generated = generate(
            args.seed + segment, args.symbols, args.functions, (depth,), count,
            forbidden, f"curriculum-{task_name}-d{depth}",
        )
        selected = [row for row in generated if row["task"] == task_name]
        if len(selected) != count:
            raise RuntimeError(
                f"Expected {count} {task_name}/d{depth} rows, got {len(selected)}"
            )
        validate(selected, args.symbols, args.functions)
        source_rows.extend(selected)
        segment_manifest.append({
            "task": task_name, "depth": depth, "examples": count,
        })

    random.Random(args.seed + 10_000).shuffle(source_rows)
    if len({row["table_id"] for row in source_rows}) != len(source_rows):
        raise RuntimeError("Curriculum contains duplicate tables")
    overlap = base_forbidden.intersection(row["table_id"] for row in source_rows)
    if overlap:
        raise RuntimeError(f"Curriculum overlaps base data by {len(overlap)} tables")
    converted = convert(source_rows, tokenizer)
    args.out.mkdir(parents=True)
    Dataset.from_list(converted).save_to_disk(str(args.out / "train"))
    manifest = {
        "version": 2,
        "task": "d4_heavy_official_elf_curriculum",
        "seed": args.seed,
        "symbols": args.symbols,
        "functions": args.functions,
        "scale": args.scale,
        "examples": len(source_rows),
        "unique_tables": len({row["table_id"] for row in source_rows}),
        "excluded_base_unique_tables": excluded_count,
        "excluded_base_split_increments": excluded_split_counts,
        "segments": segment_manifest,
        "groups": dict(sorted(Counter(
            f"{row['task']}/d{row['depth']}" for row in source_rows
        ).items())),
        "answers": dict(sorted(Counter(row["target"] for row in converted).items())),
        "source_sha256": digest(source_rows),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
