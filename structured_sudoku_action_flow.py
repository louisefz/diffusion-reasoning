#!/usr/bin/env python
"""Structured continuous action flow for verifier-executable Sudoku reasoning.

This is deliberately a small overfitting/generalization gate before another
large ELF run.  A shared board Transformer is trained either as:

* ``flow``: Gaussian -> continuous 3xN action code -> row/column/value, or
* ``classifier``: board -> row/column/value directly.

The two models have the same architecture and parameter count.  The experiment
therefore tests the structured action interface, not language formatting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch import nn

from official_sudoku_action_eval import apply_action


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class ModelConfig:
    size: int = 9
    width: int = 192
    layers: int = 4
    heads: int = 6
    dropout: float = 0.0


class StructuredActionModel(nn.Module):
    """Bidirectional board encoder plus three continuous action tokens."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        n, width = config.size, config.width
        self.current_value = nn.Embedding(n + 1, width)
        self.clue_value = nn.Embedding(n + 1, width)
        self.row = nn.Embedding(n, width)
        self.column = nn.Embedding(n, width)
        self.box = nn.Embedding(n, width)
        self.action_in = nn.ModuleList([nn.Linear(n, width) for _ in range(3)])
        self.action_type = nn.Embedding(3, width)
        self.time = nn.Sequential(
            nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width)
        )
        block = nn.TransformerEncoderLayer(
            width, config.heads, 4 * width, dropout=config.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            block, config.layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(width)
        self.action_out = nn.ModuleList([nn.Linear(width, n) for _ in range(3)])

        positions = torch.arange(n * n)
        rows = positions // n
        columns = positions % n
        box_width = int(math.isqrt(n))
        boxes = (rows // box_width) * box_width + columns // box_width
        self.register_buffer("rows", rows, persistent=False)
        self.register_buffer("columns", columns, persistent=False)
        self.register_buffer("boxes", boxes, persistent=False)

    def forward(
        self,
        puzzle: torch.Tensor,
        current: torch.Tensor,
        action_state: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        board = (
            self.current_value(current)
            + self.clue_value(puzzle)
            + self.row(self.rows)[None]
            + self.column(self.columns)[None]
            + self.box(self.boxes)[None]
        )
        phase = torch.stack(
            [time, torch.sin(math.pi * time), torch.cos(math.pi * time)], dim=-1
        )
        phase = self.time(phase)[:, None, :]
        actions = []
        for index in range(3):
            actions.append(
                self.action_in[index](action_state[:, index])
                + self.action_type.weight[index][None]
                + phase[:, 0]
            )
        hidden = self.transformer(torch.cat([board, torch.stack(actions, dim=1)], dim=1))
        hidden = self.norm(hidden[:, -3:])
        return torch.stack(
            [self.action_out[index](hidden[:, index]) for index in range(3)], dim=1
        )


def load_rows(path: str, limit: int | None = None) -> dict[str, torch.Tensor]:
    if path.endswith(".npz"):
        packed = np.load(path)
        stop = len(packed["labels"]) if limit is None else min(limit, len(packed["labels"]))
        return {
            "puzzle": torch.from_numpy(packed["puzzle"][:stop].astype(np.int64)),
            "current": torch.from_numpy(packed["current"][:stop].astype(np.int64)),
            "labels": torch.from_numpy(packed["labels"][:stop].astype(np.int64)),
        }
    dataset = load_from_disk(path)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    puzzle = torch.tensor(np.asarray(dataset["puzzle"]), dtype=torch.long)
    current = torch.tensor(np.asarray(dataset["current"]), dtype=torch.long)
    position = torch.tensor(np.asarray(dataset["branch_position"]), dtype=torch.long)
    value = torch.tensor(np.asarray(dataset["branch_value"]), dtype=torch.long) - 1
    size = int(dataset[0]["size"])
    labels = torch.stack([position // size, position % size, value], dim=1)
    return {"puzzle": puzzle, "current": current, "labels": labels}


def batch_from(data, indices, device):
    return tuple(data[key][indices].to(device, non_blocking=True)
                 for key in ("puzzle", "current", "labels"))


def encoded_action(labels: torch.Tensor, size: int) -> torch.Tensor:
    return 2.0 * F.one_hot(labels, num_classes=size).float() - 1.0


def flow_loss(model, puzzle, current, labels, generator):
    batch = len(puzzle)
    target = encoded_action(labels, model.config.size)
    noise = torch.randn(target.shape, device=target.device, generator=generator)
    time = torch.rand(batch, device=target.device, generator=generator)
    state = (1.0 - time[:, None, None]) * noise + time[:, None, None] * target
    velocity_target = target - noise
    velocity = model(puzzle, current, state, time)
    clean = state + (1.0 - time[:, None, None]) * velocity
    fm = F.mse_loss(velocity, velocity_target)
    ce = sum(F.cross_entropy(clean[:, index], labels[:, index]) for index in range(3)) / 3
    return fm + 0.5 * ce, fm.detach(), ce.detach()


def classifier_loss(model, puzzle, current, labels):
    batch, size = len(puzzle), model.config.size
    state = torch.zeros(batch, 3, size, device=puzzle.device)
    time = torch.zeros(batch, device=puzzle.device)
    logits = model(puzzle, current, state, time)
    ce = sum(F.cross_entropy(logits[:, index], labels[:, index]) for index in range(3)) / 3
    return ce, ce.detach().new_zeros(()), ce.detach()


@torch.no_grad()
def predict(model, puzzle, current, model_type, nfe, generator):
    batch, size = len(puzzle), model.config.size
    if model_type == "classifier":
        state = torch.zeros(batch, 3, size, device=puzzle.device)
        time = torch.zeros(batch, device=puzzle.device)
        return model(puzzle, current, state, time).argmax(dim=-1)
    state = torch.randn((batch, 3, size), device=puzzle.device, generator=generator)
    for step in range(nfe):
        time = torch.full((batch,), step / nfe, device=puzzle.device)
        state = state + model(puzzle, current, state, time) / nfe
    return state.argmax(dim=-1)


@torch.no_grad()
def evaluate(model, data, model_type, nfe, batch_size, device, seed,
             verifier_limit=256, evaluation_limit=None):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    predictions, labels = [], []
    total = len(data["labels"])
    if evaluation_limit is not None and evaluation_limit > 0:
        total = min(total, evaluation_limit)
    for start in range(0, total, batch_size):
        indices = slice(start, min(start + batch_size, total))
        puzzle, current, target = batch_from(data, indices, device)
        output = predict(model, puzzle, current, model_type, nfe, generator)
        predictions.append(output.cpu())
        labels.append(target.cpu())
    predictions = torch.cat(predictions)
    labels = torch.cat(labels)
    component = (predictions == labels).float().mean(dim=0)
    exact = (predictions == labels).all(dim=1).float().mean().item()

    size = model.config.size
    box = int(math.isqrt(size))
    legal = 0
    checked = min(verifier_limit, len(predictions))
    for index in range(checked):
        row, column, value = map(int, predictions[index].tolist())
        action = (row * size + column, value + 1)
        successor, details = apply_action(
            tuple(map(int, data["current"][index].tolist())),
            tuple(map(int, data["puzzle"][index].tolist())),
            action, size, box, seed + index, require_completion=False,
        )
        legal += int(successor is not None and details["legal_action"])
    return {
        "exact_target_accuracy": exact,
        "row_accuracy": float(component[0]),
        "column_accuracy": float(component[1]),
        "value_accuracy": float(component[2]),
        "local_legal_action_rate": legal / max(checked, 1),
        "verifier_examples": checked,
    }


def train_one(model_type, config, train_data, test_data, args, device, output_dir):
    seed_everything(args.seed + (0 if model_type == "flow" else 10_000))
    model = StructuredActionModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    generator = torch.Generator(device=device).manual_seed(
        args.seed + (1 if model_type == "flow" else 10_001)
    )
    index_generator = torch.Generator(device="cpu").manual_seed(
        args.seed + (2 if model_type == "flow" else 10_002)
    )
    history = []
    for step in range(1, args.steps + 1):
        model.train()
        indices = torch.randint(
            len(train_data["labels"]), (args.batch_size,), generator=index_generator
        )
        puzzle, current, labels = batch_from(train_data, indices, device)
        optimizer.zero_grad(set_to_none=True)
        if model_type == "flow":
            loss, fm, ce = flow_loss(model, puzzle, current, labels, generator)
        else:
            loss, fm, ce = classifier_loss(model, puzzle, current, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            train_metrics = evaluate(
                model, train_data, model_type, args.nfe, args.eval_batch_size,
                device, args.seed + step, verifier_limit=args.verifier_limit,
                evaluation_limit=args.evaluation_limit,
            )
            test_metrics = evaluate(
                model, test_data, model_type, args.nfe, args.eval_batch_size,
                device, args.seed + 50_000 + step, verifier_limit=args.verifier_limit,
                evaluation_limit=args.evaluation_limit,
            )
            row = {
                "model": model_type, "step": step, "loss": float(loss.detach()),
                "flow_matching": float(fm), "cross_entropy": float(ce),
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    torch.save({"config": asdict(config), "model": model.state_dict()},
               output_dir / f"{model_type}.pt")
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-limit", type=int, default=1024)
    parser.add_argument("--test-limit", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--verifier-limit", type=int, default=256)
    parser.add_argument("--evaluation-limit", type=int, default=0,
                        help="Limit examples per periodic evaluation; 0 uses all.")
    parser.add_argument("--nfe", type=int, default=16)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_data = load_rows(args.train_data, args.train_limit)
    test_data = load_rows(args.test_data, args.test_limit)
    size = int(round(math.sqrt(train_data["current"].shape[1])))
    config = ModelConfig(
        size=size, width=args.width, layers=args.layers, heads=args.heads
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"args": vars(args), "config": asdict(config),
                      "device": str(device)}, indent=2), flush=True)

    rows = []
    for model_type in ("classifier", "flow"):
        rows.extend(train_one(
            model_type, config, train_data, test_data, args, device, output_dir
        ))
    with (output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        model_type: [row for row in rows if row["model"] == model_type][-1]
        for model_type in ("classifier", "flow")
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
