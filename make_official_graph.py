"""Build controlled graph-reachability data for official ELF."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

from graph_task import generate, validate


DEFAULT_TRAIN_DEPTHS = (1, 2, 3, 4, 5)
DEFAULT_EVAL_DEPTHS = (1, 2, 3, 4, 5, 6, 8, 10, 12)


def parse_depths(value):
    depths = tuple(int(item) for item in value.split(","))
    if not depths or any(depth < 1 for depth in depths):
        raise argparse.ArgumentTypeError("depths must be positive comma-separated integers")
    return depths


def render(row, num_nodes):
    edges = " ".join(f"{left}>{right}" for left, right in row["edges"])
    mode = "REACH" if row["task"] == "reach" else "EDGE"
    return (
        f"Nodes are 0 through {num_nodes - 1}. Directed edges: {edges}. "
        f"Mode {mode}. Query {row['source']} {row['target']}. "
        "REACH asks whether any directed path exists. "
        "EDGE asks whether the direct edge is listed. Answer 0 or 1:"
    )


def convert(rows, tokenizer, num_nodes):
    converted = []
    for row in rows:
        prompt = render(row, num_nodes)
        converted.append({
            "condition_input_ids": tokenizer(prompt, add_special_tokens=False)["input_ids"],
            "input_ids": tokenizer(str(row["answer"]), add_special_tokens=False)["input_ids"],
            "input": prompt,
            "target": str(row["answer"]),
            "task": row["task"],
            "depth": row["depth"],
            "episode_id": row["episode_id"],
            "graph_id": row["graph_id"],
            "source": row["source"],
            "query_target": row["target"],
            "edges": row["edges"],
            "states": row["states"],
        })
    return converted


def digest(rows):
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    return hashlib.sha256(payload.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--nodes", type=int, default=16)
    parser.add_argument("--train-depths", type=parse_depths, default=DEFAULT_TRAIN_DEPTHS)
    parser.add_argument("--eval-depths", type=parse_depths, default=DEFAULT_EVAL_DEPTHS)
    parser.add_argument("--train-episodes", type=int, default=200_000)
    parser.add_argument("--eval-episodes", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"Refusing to overwrite {args.out}")
    if args.nodes <= max(args.train_depths + args.eval_depths):
        parser.error("nodes must exceed the largest requested shortest-path depth")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    args.out.mkdir(parents=True)
    seen_graphs = set()
    manifest = {
        "version": 1, "task": "directed_graph_reachability",
        "seed": args.seed, "nodes": args.nodes,
        "train_depths": list(args.train_depths),
        "eval_depths": list(args.eval_depths), "splits": {},
    }
    for offset, split in enumerate(("train", "validation", "test")):
        episodes = args.train_episodes if split == "train" else args.eval_episodes
        depths = args.train_depths if split == "train" else args.eval_depths
        source = generate(args.seed + offset, args.nodes, depths, episodes, seen_graphs, split)
        validate(source, args.nodes)
        converted = convert(source, tokenizer, args.nodes)
        Dataset.from_list(converted).save_to_disk(str(args.out / split))
        lengths = [len(row["condition_input_ids"]) for row in converted]
        manifest["splits"][split] = {
            "episodes": episodes, "examples": len(converted),
            "source_sha256": digest(source),
            "unique_graphs": len({row["graph_id"] for row in source}),
            "groups": dict(sorted(Counter(
                f"{row['task']}/d{row['depth']}" for row in source).items())),
            "answers_by_group": dict(sorted(Counter(
                f"{row['task']}/d{row['depth']}/y{row['answer']}" for row in source
            ).items())),
            "condition_tokens_min": min(lengths),
            "condition_tokens_max": max(lengths),
        }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
