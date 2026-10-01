#!/usr/bin/env python
"""Pack only the tensors needed by structured action experiments."""

import argparse
from pathlib import Path

import numpy as np
from datasets import load_from_disk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        dataset = load_from_disk(str(args.source / split))
        puzzle = np.asarray(dataset["puzzle"], dtype=np.uint8)
        current = np.asarray(dataset["current"], dtype=np.uint8)
        position = np.asarray(dataset["branch_position"], dtype=np.uint8)
        value = np.asarray(dataset["branch_value"], dtype=np.uint8) - 1
        size = int(dataset[0]["size"])
        labels = np.stack([position // size, position % size, value], axis=1).astype(np.uint8)
        destination = args.output / f"{split}.npz"
        np.savez_compressed(destination, puzzle=puzzle, current=current, labels=labels)
        print(f"{split}: {len(labels)} rows -> {destination}", flush=True)


if __name__ == "__main__":
    main()
