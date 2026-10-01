"""Create a fixed 256-example subset for an official ELF overfit test."""

import argparse
from collections import defaultdict
from pathlib import Path

from datasets import load_from_disk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--stratify", default=None,
                        help="Comma-separated columns for an equal-size stratified subset")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error(f"Refusing to overwrite {output}")
    dataset = load_from_disk(args.source)
    if args.stratify:
        columns = tuple(part.strip() for part in args.stratify.split(",") if part.strip())
        groups = defaultdict(list)
        for index, row in enumerate(dataset):
            groups[tuple(row[column] for column in columns)].append(index)
        if args.count % len(groups):
            parser.error(f"count={args.count} is not divisible by {len(groups)} groups")
        per_group = args.count // len(groups)
        if min(map(len, groups.values())) < per_group:
            parser.error(f"At least one group has fewer than {per_group} examples")
        indices = sorted(index for values in groups.values() for index in values[:per_group])
        subset = dataset.select(indices)
    else:
        subset = dataset.select(range(args.count))
    subset.save_to_disk(str(output))
    print(f"saved={output} examples={len(subset)} unique_tables={len(set(subset['table_id']))}")


if __name__ == "__main__":
    main()
