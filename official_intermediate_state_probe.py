#!/usr/bin/env python
"""Probe ground-truth d4 composition states across ELF-B denoiser blocks."""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent
OFFICIAL_SRC = ROOT / "official-elf" / "src"
sys.path.insert(0, str(OFFICIAL_SRC))

from configs.config import load_config_from_yaml
from modules.t5_encoder import get_encoder
from official_diagnostics import load_model
from official_layerwise_probe import extract_features
from utils.data_utils import load_dataset_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--eval-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--alpha", action="append", type=float, default=None)
    return parser.parse_args()


def select_compose_d4(dataset, count, seed):
    table = dataset.data.table
    mask = pc.and_(
        pc.equal(table.column("task"), pa.scalar("compose")),
        pc.equal(table.column("depth"), pa.scalar(4)),
    )
    candidates = pc.indices_nonzero(mask).to_numpy().tolist()
    if len(candidates) < count:
        raise ValueError(f"Requested {count} compose/d4, found {len(candidates)}")
    chosen = sorted(random.Random(seed).sample(candidates, count))
    return dataset.select(chosen), chosen


def train_state_probes(train_features, train_states, eval_features, eval_states,
                       alphas, seed):
    fit_indices, dev_indices = train_test_split(
        np.arange(len(train_states)), test_size=0.2, random_state=seed,
        stratify=train_states[:, -1],
    )
    rows = []
    for state_step in range(1, 5):
        train_labels = train_states[:, state_step]
        eval_labels = eval_states[:, state_step]
        for layer in range(train_features.shape[1]):
            features = train_features[:, layer]
            best = None
            for alpha in alphas:
                probe = make_pipeline(
                    StandardScaler(), RidgeClassifier(alpha=alpha),
                )
                probe.fit(features[fit_indices], train_labels[fit_indices])
                dev_accuracy = accuracy_score(
                    train_labels[dev_indices], probe.predict(features[dev_indices]),
                )
                candidate = (dev_accuracy, -alpha, alpha)
                if best is None or candidate > best:
                    best = candidate
            probe = make_pipeline(
                StandardScaler(), RidgeClassifier(alpha=best[2]),
            )
            probe.fit(features, train_labels)
            eval_accuracy = accuracy_score(
                eval_labels, probe.predict(eval_features[:, layer]),
            )
            rows.append({
                "state_step": state_step,
                "layer": layer,
                "layer_name": "input_projection" if layer == 0 else f"block_{layer}",
                "alpha": best[2],
                "dev_accuracy": best[0],
                "eval_accuracy": eval_accuracy,
            })
        print(
            f"State s{state_step}: best eval="
            f"{max(row['eval_accuracy'] for row in rows if row['state_step'] == state_step):.4f}",
            flush=True,
        )
    return rows


def _first_layer(rows, step, threshold):
    for row in rows:
        if row["state_step"] == step and row["eval_accuracy"] >= threshold:
            return row["layer"]
    return None


def _plot(rows, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray([
        [next(row["eval_accuracy"] for row in rows
              if row["state_step"] == step and row["layer"] == layer)
         for layer in range(13)]
        for step in range(1, 5)
    ])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for step in range(1, 5):
        axes[0].plot(range(13), matrix[step - 1], marker="o", label=f"state s{step}")
    axes[0].axhline(0.125, color="black", linestyle=":", linewidth=1)
    axes[0].axhline(0.9, color="gray", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Representation layer")
    axes[0].set_ylabel("Held-out probe accuracy")
    axes[0].set_xticks(range(13))
    axes[0].set_ylim(0, 1.03)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    image = axes[1].imshow(matrix, aspect="auto", vmin=0.125, vmax=1.0, cmap="viridis")
    axes[1].set_xlabel("Representation layer")
    axes[1].set_ylabel("Ground-truth composition state")
    axes[1].set_xticks(range(13))
    axes[1].set_yticks(range(4), ["s1", "s2", "s3", "s4 (answer)"])
    fig.colorbar(image, ax=axes[1], label="Accuracy")
    fig.tight_layout()
    fig.savefig(output_dir / "intermediate_state_probe.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config_from_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name or config.encoder_model_name
    )
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    model, checkpoint_step = load_model(
        config, args.checkpoint, encoder_config, tokenizer, device,
    )
    train_dataset = load_dataset_split(config.data_path)
    eval_dataset = load_dataset_split(config.eval_data_path)
    train_selected, train_ids = select_compose_d4(
        train_dataset, args.train_samples, args.seed,
    )
    eval_selected, eval_ids = select_compose_d4(
        eval_dataset, args.eval_samples, args.seed + 1,
    )
    train = extract_features(
        train_selected, ["compose/d4"] * len(train_selected), model, encoder,
        tokenizer, config, device, args.batch_size, args.seed + 2,
    )
    evaluation = extract_features(
        eval_selected, ["compose/d4"] * len(eval_selected), model, encoder,
        tokenizer, config, device, args.batch_size, args.seed + 3,
    )
    train_states = np.stack(train_selected["states"]).astype(np.int64)
    eval_states = np.stack(eval_selected["states"]).astype(np.int64)
    if train_states.shape[1] != 5 or eval_states.shape[1] != 5:
        raise ValueError("compose/d4 must have [start, s1, s2, s3, s4]")
    alphas = args.alpha or [0.1, 1.0, 10.0, 100.0]
    rows = train_state_probes(
        train["features"], train_states, evaluation["features"], eval_states,
        alphas, args.seed,
    )
    emergence = {
        f"s{step}": {
            "accuracy_50": _first_layer(rows, step, 0.5),
            "accuracy_80": _first_layer(rows, step, 0.8),
            "accuracy_90": _first_layer(rows, step, 0.9),
        }
        for step in range(1, 5)
    }
    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": checkpoint_step,
        "device": str(device),
        "train_samples": args.train_samples,
        "eval_samples": args.eval_samples,
        "train_source_ids": train_ids,
        "eval_source_ids": eval_ids,
        "alphas": alphas,
        "state_definition": "s_k is the value after applying the kth function in compose/d4",
        "emergence_layers": emergence,
        "metrics": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    with (output_dir / "probe_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, output_dir)
    print(json.dumps(emergence, indent=2), flush=True)
    print(f"Saved intermediate-state artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
