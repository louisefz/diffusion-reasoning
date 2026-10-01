#!/usr/bin/env python
"""Create a large, training-disjoint graph split for mechanistic experiments."""

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

from graph_task import generate, validate
from make_official_graph import convert, digest, parse_depths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--depths", type=parse_depths, default=(4, 8, 12))
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--nodes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"Refusing to overwrite {args.out}")

    seen = set()
    for split in ("train", "validation", "test"):
        dataset = load_from_disk(str(args.reference / split))
        seen.update(map(str, dataset["graph_id"]))
    reference_ids = set(seen)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    rows = generate(
        args.seed, args.nodes, args.depths, args.episodes, seen, "mechanism"
    )
    validate(rows, args.nodes)
    converted = convert(rows, tokenizer, args.nodes)
    graph_ids = {row["graph_id"] for row in converted}
    assert not graph_ids & reference_ids
    Dataset.from_list(converted).save_to_disk(str(args.out))

    manifest = {
        "version": 1,
        "seed": args.seed,
        "nodes": args.nodes,
        "depths": list(args.depths),
        "episodes": args.episodes,
        "examples": len(converted),
        "unique_graphs": len(graph_ids),
        "reference_overlap": len(graph_ids & reference_ids),
        "groups": dict(sorted(Counter(
            f"{row['task']}/d{row['depth']}/y{row['target']}" for row in converted
        ).items())),
        "sha256": digest(rows),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
