"""Unit test for repeatable computation with a conditional flow operator.

The experiment deliberately separates two clocks:

* K: calls to the learned one-hop BFS transition (algorithmic depth)
* M: Euler function evaluations used to realize one conditional flow transition

Training uses ground-truth transitions from only the first ``train_depth`` BFS
rounds. Evaluation rolls the same weight-tied operator out for as many as 32
rounds.  A parameter-matched recurrent operator is the primary control.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def closure_step(adjacency: torch.Tensor, belief: torch.Tensor) -> torch.Tensor:
    """One cumulative directed-reachability update.

    adjacency[b, u, v] is one exactly when u -> v. Beliefs are binary.
    """
    incoming = torch.bmm(belief.unsqueeze(1), adjacency).squeeze(1)
    return torch.logical_or(belief.bool(), incoming > 0).float()


def layered_graph_batch(
    batch_size: int,
    num_nodes: int,
    chain_depth: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Graphs with a certified long path and permuted node identities.

    The remaining nodes form distractor subgraphs and may point *into* the
    reachable chain. They never receive edges from it, so they stay unreachable.
    """
    if chain_depth + 1 > num_nodes:
        raise ValueError("chain_depth + 1 must not exceed num_nodes")
    adjacency = torch.zeros(batch_size, num_nodes, num_nodes, device=device)
    sources = torch.zeros(batch_size, num_nodes, device=device)
    for batch_index in range(batch_size):
        permutation = torch.randperm(num_nodes, generator=generator, device=device)
        chain = permutation[: chain_depth + 1]
        distractors = permutation[chain_depth + 1 :]
        adjacency[batch_index, chain[:-1], chain[1:]] = 1.0

        # Add harmless edges among distractors and from distractors into the
        # chain. This prevents a bare path-position lookup while preserving the
        # certified distance of every chain node.
        if distractors.numel() > 1:
            count = max(1, int(0.08 * distractors.numel() ** 2))
            left = distractors[
                torch.randint(distractors.numel(), (count,), generator=generator, device=device)
            ]
            right = distractors[
                torch.randint(distractors.numel(), (count,), generator=generator, device=device)
            ]
            keep = left != right
            adjacency[batch_index, left[keep], right[keep]] = 1.0
            into_count = max(1, distractors.numel() // 3)
            left = distractors[
                torch.randint(distractors.numel(), (into_count,), generator=generator, device=device)
            ]
            right = chain[
                torch.randint(chain.numel(), (into_count,), generator=generator, device=device)
            ]
            adjacency[batch_index, left, right] = 1.0
        sources[batch_index, chain[0]] = 1.0
    return adjacency, sources


@torch.no_grad()
def advance(adjacency: torch.Tensor, belief: torch.Tensor, steps: int) -> torch.Tensor:
    for _ in range(steps):
        belief = closure_step(adjacency, belief)
    return belief


def training_batch(
    batch_size: int,
    num_nodes: int,
    train_depth: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Graphs can be longer than the supervised rollout, but k is restricted to
    # the first train_depth rounds.
    chain_depth = max(train_depth + 2, 8)
    adjacency, source = layered_graph_batch(
        batch_size, num_nodes, chain_depth, device, generator
    )
    k = torch.randint(
        0, train_depth, (batch_size,), generator=generator, device=device
    )
    states = [source]
    for _ in range(train_depth):
        states.append(closure_step(adjacency, states[-1]))
    stacked = torch.stack(states, dim=1)
    rows = torch.arange(batch_size, device=device)
    current = stacked[rows, k]
    target = stacked[rows, k + 1]
    return adjacency, current, target


class LocalGraphOperator(nn.Module):
    """A single-hop message-passing network shared by both methods."""

    def __init__(self, width: int = 128):
        super().__init__()
        self.width = width
        self.node_encoder = nn.Sequential(
            nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.message = nn.Sequential(nn.Linear(width, width), nn.SiLU())
        self.update = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )

    def forward(
        self,
        adjacency: torch.Tensor,
        belief: torch.Tensor,
        state: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_nodes = belief.shape
        time_feature = time.reshape(batch_size, 1).expand(batch_size, num_nodes)
        features = torch.stack([belief, state, time_feature], dim=-1)
        hidden = self.node_encoder(features)
        messages = torch.bmm(adjacency.transpose(1, 2), self.message(hidden))
        # Bound degree-dependent scale while retaining whether a predecessor
        # sent a message.
        degree = adjacency.transpose(1, 2).sum(dim=-1, keepdim=True).clamp_min(1.0)
        messages = messages / degree.sqrt()
        return self.update(torch.cat([hidden, messages], dim=-1)).squeeze(-1)


def encode_binary(value: torch.Tensor) -> torch.Tensor:
    return value.mul(2.0).sub(1.0)


def flow_loss(
    model: nn.Module,
    adjacency: torch.Tensor,
    current: torch.Tensor,
    target: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch_size = current.shape[0]
    noise = torch.randn(
        target.shape, generator=generator, device=target.device, dtype=target.dtype
    )
    time = torch.rand(batch_size, generator=generator, device=target.device)
    encoded_target = encode_binary(target)
    mixed = (1.0 - time[:, None]) * noise + time[:, None] * encoded_target
    target_velocity = encoded_target - noise
    predicted_velocity = model(adjacency, current, mixed, time)
    denoised = mixed + (1.0 - time[:, None]) * predicted_velocity
    matching = F.mse_loss(predicted_velocity, target_velocity)
    decoding = F.binary_cross_entropy_with_logits(2.0 * denoised, target)
    loss = matching + 0.25 * decoding
    exact = ((denoised > 0) == target.bool()).all(dim=1).float().mean()
    return loss, {
        "flow_matching": float(matching.detach()),
        "denoise_bce": float(decoding.detach()),
        "train_exact": float(exact.detach()),
    }


def recurrent_loss(
    model: nn.Module,
    adjacency: torch.Tensor,
    current: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    time = torch.zeros(current.shape[0], device=current.device)
    logits = model(adjacency, current, current, time)
    loss = F.binary_cross_entropy_with_logits(logits, target)
    exact = ((logits > 0) == target.bool()).all(dim=1).float().mean()
    return loss, {"transition_bce": float(loss.detach()), "train_exact": float(exact.detach())}


@torch.no_grad()
def flow_transition(
    model: nn.Module,
    adjacency: torch.Tensor,
    current: torch.Tensor,
    nfe: int,
    generator: torch.Generator,
) -> torch.Tensor:
    state = torch.randn(
        current.shape, generator=generator, device=current.device, dtype=current.dtype
    )
    step_size = 1.0 / nfe
    for index in range(nfe):
        time = torch.full(
            (current.shape[0],), index / nfe, device=current.device
        )
        state = state + step_size * model(adjacency, current, state, time)
    return (state > 0).float()


@torch.no_grad()
def recurrent_transition(
    model: nn.Module, adjacency: torch.Tensor, current: torch.Tensor
) -> torch.Tensor:
    time = torch.zeros(current.shape[0], device=current.device)
    return (model(adjacency, current, current, time) > 0).float()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    model_type: str,
    num_nodes: int,
    batches: int,
    batch_size: int,
    nfe_values: list[int],
    rollout_values: list[int],
    seed: int,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    model.eval()
    rows: list[dict[str, float | int | str]] = []
    max_rollout = max(rollout_values)
    for nfe in nfe_values if model_type == "flow" else [1]:
        correct_by_k = {k: 0 for k in rollout_values}
        node_correct_by_k = {k: 0.0 for k in rollout_values}
        transition_correct = 0
        transition_total = 0
        examples = 0
        generator = torch.Generator(device=device).manual_seed(seed + 1000 * nfe)
        for _ in range(batches):
            adjacency, truth = layered_graph_batch(
                batch_size, num_nodes, max_rollout, device, generator
            )
            prediction = truth.clone()
            for step in range(1, max_rollout + 1):
                expected_local = closure_step(adjacency, prediction)
                if model_type == "flow":
                    prediction = flow_transition(
                        model, adjacency, prediction, nfe, generator
                    )
                else:
                    prediction = recurrent_transition(model, adjacency, prediction)
                transition_correct += int(
                    (prediction == expected_local).all(dim=1).sum().item()
                )
                transition_total += batch_size
                truth = closure_step(adjacency, truth)
                if step in correct_by_k:
                    correct_by_k[step] += int(
                        (prediction == truth).all(dim=1).sum().item()
                    )
                    node_correct_by_k[step] += float(
                        (prediction == truth).float().mean(dim=1).sum().item()
                    )
            examples += batch_size
        for rollout in rollout_values:
            rows.append(
                {
                    "model": model_type,
                    "nfe_per_transition": nfe,
                    "reasoning_rounds": rollout,
                    "total_model_calls": nfe * rollout if model_type == "flow" else rollout,
                    "state_exact_accuracy": correct_by_k[rollout] / examples,
                    "node_accuracy": node_correct_by_k[rollout] / examples,
                    "local_transition_consistency": transition_correct / transition_total,
                    "examples": examples,
                }
            )
    return rows


def save_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["flow", "recurrent"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--num-nodes", type=int, default=40)
    parser.add_argument("--train-depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2)

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project="latent-diffusion-reasoning",
            entity="nlp_louise-org",
            name=f"bfs-transition-{args.model}-{os.environ.get('SLURM_JOB_ID', 'local')}",
            config=vars(args),
        )

    model = LocalGraphOperator(args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(args.seed + 17)
    history: list[dict[str, float | int]] = []
    model.train()
    for step in range(1, args.steps + 1):
        adjacency, current, target = training_batch(
            args.batch_size,
            args.num_nodes,
            args.train_depth,
            device,
            generator,
        )
        if args.model == "flow":
            loss, metrics = flow_loss(model, adjacency, current, target, generator)
        else:
            loss, metrics = recurrent_loss(model, adjacency, current, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record = {"step": step, "loss": float(loss.detach()), **metrics}
            history.append(record)
            print(json.dumps(record), flush=True)
            if wandb_run is not None:
                wandb.log(record, step=step)

    checkpoint = {
        "model": model.state_dict(),
        "args": vars(args),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    torch.save(checkpoint, output_dir / "checkpoint.pt")
    with (output_dir / "train_history.json").open("w") as handle:
        json.dump(history, handle, indent=2)

    rows = evaluate(
        model,
        args.model,
        args.num_nodes,
        args.eval_batches,
        args.eval_batch_size,
        [1, 2, 4, 8],
        [4, 8, 16, 32],
        args.seed + 99,
        device,
    )
    save_rows(output_dir / "evaluation.csv", rows)
    summary = {
        "parameter_count": checkpoint["parameter_count"],
        "model": args.model,
        "training_depth": args.train_depth,
        "evaluation": rows,
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    if wandb_run is not None:
        try:
            table = wandb.Table(dataframe=None, columns=list(rows[0]))
            for row in rows:
                table.add_data(*[row[column] for column in rows[0]])
            wandb.log({"depth_extrapolation": table})
        except (OSError, PermissionError) as error:
            # Evaluation artifacts are already persisted as CSV/JSON. Logging
            # must never turn a successful scientific run into a failed job.
            print(f"W&B table logging skipped: {error}", flush=True)
        finally:
            wandb.finish()


if __name__ == "__main__":
    main()
