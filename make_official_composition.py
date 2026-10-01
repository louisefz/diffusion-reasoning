"""Build paired function-composition and lookup-control data for official ELF."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

from task import generate, validate


TRAIN_DEPTHS = (1, 2, 4, 8)
EVAL_DEPTHS = (1, 2, 4, 8, 12, 16)


def render(row):
    """Render a compact, unambiguous prompt while keeping paired inputs matched."""
    n = len(row["tables"][0])
    tables = "; ".join(
        f"F{i}: " + " ".join(map(str, table))
        for i, table in enumerate(row["tables"])
    )
    program = " ".join(f"F{i}" for i in row["program"])
    mode = "FULL" if row["task"] == "compose" else "FIRST"
    return (
        f"Each function list gives outputs for inputs 0 through {n - 1}. "
        f"{tables}. Mode {mode}. Start {row['start']}. Program {program}. "
        "FULL applies every function; FIRST applies only the first function. Answer:"
    )


def convert(rows, tokenizer):
    converted = []
    for row in rows:
        prompt = render(row)
        converted.append({
            "condition_input_ids": tokenizer(prompt, add_special_tokens=False)["input_ids"],
            "input_ids": tokenizer(str(row["answer"]), add_special_tokens=False)["input_ids"],
            "input": prompt,
            "target": str(row["answer"]),
            "task": row["task"],
            "depth": row["depth"],
            "episode_id": row["episode_id"],
            "table_id": row["table_id"],
            "start": row["start"],
            "program": row["program"],
            "states": row["states"],
        })
    return converted


def rows_digest(rows):
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    return hashlib.sha256(payload.encode()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--symbols", type=int, default=8)
    p.add_argument("--functions", type=int, default=4)
    p.add_argument("--train-episodes", type=int, default=50_000,
                   help="Each episode yields one composition and one lookup row.")
    p.add_argument("--eval-episodes", type=int, default=1_200)
    p.add_argument("--seed", type=int, default=20260909)
    args = p.parse_args()
    if args.out.exists():
        p.error(f"Refusing to overwrite {args.out}")
    if args.symbols < 3 or args.functions < 2:
        p.error("Use at least 3 symbols and 2 functions")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    args.out.mkdir(parents=True)
    seen = set()
    manifest = {
        "version": 1,
        "task": "paired_random_function_composition",
        "seed": args.seed,
        "symbols": args.symbols,
        "functions": args.functions,
        "train_depths": list(TRAIN_DEPTHS),
        "eval_depths": list(EVAL_DEPTHS),
        "splits": {},
    }
    for offset, split in enumerate(("train", "validation", "test")):
        depths = TRAIN_DEPTHS if split == "train" else EVAL_DEPTHS
        episodes = args.train_episodes if split == "train" else args.eval_episodes
        source = generate(args.seed + offset, args.symbols, args.functions,
                          depths, episodes, seen, split)
        validate(source, args.symbols, args.functions)
        rows = convert(source, tokenizer)
        Dataset.from_list(rows).save_to_disk(str(args.out / split))
        lengths = [len(row["condition_input_ids"]) for row in rows]
        manifest["splits"][split] = {
            "episodes": episodes,
            "examples": len(rows),
            "source_sha256": rows_digest(source),
            "unique_tables": len({row["table_id"] for row in rows}),
            "groups": dict(sorted(Counter(
                f"{row['task']}/d{row['depth']}" for row in rows).items())),
            "answers": dict(sorted(Counter(row["target"] for row in rows).items())),
            "condition_tokens_min": min(lengths),
            "condition_tokens_max": max(lengths),
        }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
