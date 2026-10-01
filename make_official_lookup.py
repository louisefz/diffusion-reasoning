"""Build table-disjoint, explicit key-value lookup data for official ELF."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random

from datasets import Dataset
from transformers import AutoTokenizer


def table_id(table):
    return hashlib.sha256(bytes(table)).hexdigest()


def render(table, start):
    pairs = "; ".join(f"{key} maps to {value}" for key, value in enumerate(table))
    return f"Use the table to answer the query. Table F: {pairs}. Query: F({start}). Answer:"


def make_rows(tables, count, rng, tokenizer):
    universe = [(table, start) for table in tables for start in range(len(table))]
    rng.shuffle(universe)
    if count > len(universe):
        raise ValueError(f"Requested {count} unique queries from only {len(universe)} possibilities")
    rows = []
    for table, start in universe[:count]:
        prompt, answer = render(table, start), str(table[start])
        rows.append({
            "condition_input_ids": tokenizer(prompt, add_special_tokens=False)["input_ids"],
            "input_ids": tokenizer(answer, add_special_tokens=False)["input_ids"],
            "input": prompt,
            "target": answer,
            "table_id": table_id(table),
            "start": start,
            "answer": table[start],
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--symbols", type=int, default=8)
    p.add_argument("--train", type=int, default=100_000)
    p.add_argument("--validation", type=int, default=2_000)
    p.add_argument("--test", type=int, default=2_000)
    p.add_argument("--seed", type=int, default=20260908)
    args = p.parse_args()
    if args.out.exists():
        p.error("Refusing to overwrite output directory")
    if args.symbols > 255:
        p.error("symbols must fit in one byte for stable hashes")

    rng = random.Random(args.seed)
    tables = list(itertools.permutations(range(args.symbols)))
    rng.shuffle(tables)
    # Allocate enough whole tables for each held-out split; no table crosses splits.
    n_val_tables = (args.validation + args.symbols - 1) // args.symbols
    n_test_tables = (args.test + args.symbols - 1) // args.symbols
    val_tables = tables[:n_val_tables]
    test_tables = tables[n_val_tables:n_val_tables+n_test_tables]
    train_tables = tables[n_val_tables+n_test_tables:]
    if args.train > len(train_tables) * args.symbols:
        p.error("Too many unique train queries for table-disjoint split")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    split_tables = {"train": train_tables, "validation": val_tables, "test": test_tables}
    split_counts = {"train": args.train, "validation": args.validation, "test": args.test}
    args.out.mkdir(parents=True)
    manifest = {"version": 1, "task": "explicit_single_table_lookup",
                "symbols": args.symbols, "seed": args.seed, "tokenizer": args.tokenizer,
                "splits": {}}
    for index, name in enumerate(("train", "validation", "test")):
        rows = make_rows(split_tables[name], split_counts[name], random.Random(args.seed+index+1), tokenizer)
        path = args.out/name
        Dataset.from_list(rows).save_to_disk(str(path))
        lengths = [len(r["condition_input_ids"]) + len(r["input_ids"]) for r in rows]
        ids = {r["table_id"] for r in rows}
        manifest["splits"][name] = {"examples": len(rows), "tables": len(ids),
                                     "max_total_tokens": max(lengths),
                                     "table_ids_sha256": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()}
    id_sets = []
    for name in ("train", "validation", "test"):
        ds = Dataset.load_from_disk(str(args.out/name))
        id_sets.append(set(ds["table_id"]))
    assert not (id_sets[0] & id_sets[1] or id_sets[0] & id_sets[2] or id_sets[1] & id_sets[2])
    (args.out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
