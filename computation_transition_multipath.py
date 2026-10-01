"""Multimodal gate for flow-based computation: layered multi-path routing.

Every outer call chooses one next node. Multiple outgoing choices are valid and
lead to the goal. A conditional flow operator is compared with a stochastic
categorical recurrent operator using the same one-hop backbone and supervision.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from computation_transition_bfs import LocalGraphOperator, seed_everything


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def layered_route_batch(
    batch_size: int,
    depth: int,
    width: int,
    max_nodes: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[torch.Tensor]]]:
    required = 2 + width * (depth - 1)
    if required > max_nodes:
        raise ValueError(f"Need {required} nodes for depth={depth}, max={max_nodes}")
    adjacency = torch.zeros(batch_size, max_nodes, max_nodes, device=device)
    source = torch.zeros(batch_size, max_nodes, device=device)
    goal = torch.zeros(batch_size, max_nodes, device=device)
    all_layers: list[list[torch.Tensor]] = []
    for batch_index in range(batch_size):
        permutation = torch.randperm(max_nodes, generator=generator, device=device)
        cursor = 0
        layers: list[torch.Tensor] = [permutation[cursor : cursor + 1]]
        cursor += 1
        for _ in range(depth - 1):
            layers.append(permutation[cursor : cursor + width])
            cursor += width
        layers.append(permutation[cursor : cursor + 1])

        for left_layer, right_layer in zip(layers[:-1], layers[1:]):
            # Each left node has at least two choices where possible.
            choices = min(2, right_layer.numel())
            for left in left_layer:
                order = torch.randperm(
                    right_layer.numel(), generator=generator, device=device
                )[:choices]
                adjacency[batch_index, left, right_layer[order]] = 1.0
            # Every right node remains on at least one source-goal path.
            for right in right_layer:
                parents = left_layer[
                    torch.randint(
                        left_layer.numel(), (1,), generator=generator, device=device
                    )
                ]
                adjacency[batch_index, parents, right] = 1.0

        source[batch_index, layers[0]] = 1.0
        goal[batch_index, layers[-1]] = 1.0
        all_layers.append(layers)
    return adjacency, source, goal, all_layers


def sample_next(
    adjacency: torch.Tensor,
    current: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    batch_size, num_nodes = current.shape
    current_index = current.argmax(dim=1)
    row = adjacency[torch.arange(batch_size, device=current.device), current_index]
    # Training/evaluation generators only call this on nodes with outgoing edges.
    probabilities = row / row.sum(dim=1, keepdim=True).clamp_min(1.0)
    next_index = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
    return F.one_hot(next_index, num_classes=num_nodes).float()


def training_batch(
    batch_size: int,
    train_depth: int,
    width: int,
    max_nodes: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    adjacency, current, _, _ = layered_route_batch(
        batch_size, train_depth, width, max_nodes, device, generator
    )
    layer = torch.randint(
        0, train_depth, (batch_size,), generator=generator, device=device
    )
    target = torch.empty_like(current)
    # Roll a legal prefix and select a random supervised layer per graph.
    states = [current]
    for _ in range(train_depth):
        states.append(sample_next(adjacency, states[-1], generator))
    stacked = torch.stack(states, dim=1)
    rows = torch.arange(batch_size, device=device)
    current = stacked[rows, layer]
    target = sample_next(adjacency, current, generator)
    return adjacency, current, target


def encode_choice(target: torch.Tensor) -> torch.Tensor:
    return 2.0 * target - 1.0


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
    encoded_target = encode_choice(target)
    mixed = (1.0 - time[:, None]) * noise + time[:, None] * encoded_target
    velocity_target = encoded_target - noise
    velocity = model(adjacency, current, mixed, time)
    denoised = mixed + (1.0 - time[:, None]) * velocity
    matching = F.mse_loss(velocity, velocity_target)
    choice = target.argmax(dim=1)
    classification = F.cross_entropy(denoised, choice)
    loss = matching + 0.25 * classification
    accuracy = (denoised.argmax(dim=1) == choice).float().mean()
    return loss, {
        "flow_matching": float(matching.detach()),
        "choice_ce": float(classification.detach()),
        "train_choice_accuracy": float(accuracy.detach()),
    }


def recurrent_loss(
    model: nn.Module,
    adjacency: torch.Tensor,
    current: torch.Tensor,
    target: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float]]:
    # The recurrent baseline receives random input and is trained as a proper
    # categorical policy, so it is not disadvantaged in diversity comparisons.
    noise = torch.randn(
        current.shape, generator=generator, device=current.device, dtype=current.dtype
    )
    time = torch.zeros(current.shape[0], device=current.device)
    logits = model(adjacency, current, noise, time)
    choice = target.argmax(dim=1)
    loss = F.cross_entropy(logits, choice)
    accuracy = (logits.argmax(dim=1) == choice).float().mean()
    return loss, {"choice_ce": float(loss.detach()), "train_choice_accuracy": float(accuracy.detach())}


@torch.no_grad()
def flow_choice(
    model: nn.Module,
    adjacency: torch.Tensor,
    current: torch.Tensor,
    nfe: int,
    generator: torch.Generator,
) -> torch.Tensor:
    state = torch.randn(
        current.shape, generator=generator, device=current.device, dtype=current.dtype
    )
    for index in range(nfe):
        time = torch.full((current.shape[0],), index / nfe, device=current.device)
        state = state + model(adjacency, current, state, time) / nfe
    return F.one_hot(state.argmax(dim=1), num_classes=current.shape[1]).float()


@torch.no_grad()
def recurrent_choice(
    model: nn.Module,
    adjacency: torch.Tensor,
    current: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    noise = torch.randn(
        current.shape, generator=generator, device=current.device, dtype=current.dtype
    )
    time = torch.zeros(current.shape[0], device=current.device)
    logits = model(adjacency, current, noise, time)
    probabilities = torch.softmax(logits, dim=1)
    index = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
    return F.one_hot(index, num_classes=current.shape[1]).float()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    model_type: str,
    depths: list[int],
    nfe_values: list[int],
    graph_count: int,
    restarts: int,
    width: int,
    max_nodes: int,
    device: torch.device,
    seed: int,
) -> list[dict[str, float | int | str]]:
    model.eval()
    rows: list[dict[str, float | int | str]] = []
    for depth in depths:
        for nfe in nfe_values if model_type == "flow" else [1]:
            generator = make_generator(device, seed + depth * 100 + nfe)
            adjacency, source, goal, _ = layered_route_batch(
                graph_count, depth, width, max_nodes, device, generator
            )
            adjacency = adjacency.repeat_interleave(restarts, dim=0)
            current = source.repeat_interleave(restarts, dim=0)
            goal_expanded = goal.repeat_interleave(restarts, dim=0)
            batch = current.shape[0]
            valid_steps = torch.ones(batch, dtype=torch.bool, device=device)
            paths = [current.argmax(dim=1)]
            for _ in range(depth):
                previous = current.argmax(dim=1)
                if model_type == "flow":
                    current = flow_choice(model, adjacency, current, nfe, generator)
                else:
                    current = recurrent_choice(model, adjacency, current, generator)
                chosen = current.argmax(dim=1)
                valid_steps &= adjacency[
                    torch.arange(batch, device=device), previous, chosen
                ].bool()
                paths.append(chosen)
            success = valid_steps & (current * goal_expanded).sum(dim=1).bool()
            path_matrix = torch.stack(paths, dim=1).cpu().numpy()
            successes = success.cpu().numpy()
            unique_counts = []
            successful_counts = []
            for graph_index in range(graph_count):
                start = graph_index * restarts
                end = start + restarts
                valid_paths = path_matrix[start:end][successes[start:end]]
                successful_counts.append(len(valid_paths))
                unique_counts.append(
                    len({tuple(path.tolist()) for path in valid_paths})
                )
            rows.append(
                {
                    "model": model_type,
                    "depth": depth,
                    "nfe_per_transition": nfe,
                    "total_model_calls": depth * nfe if model_type == "flow" else depth,
                    "path_success_rate": float(success.float().mean().item()),
                    "edge_valid_rate": float(valid_steps.float().mean().item()),
                    "mean_unique_successful_paths": float(np.mean(unique_counts)),
                    "unique_fraction_of_restarts": float(np.mean(unique_counts) / restarts),
                    "mean_successful_samples": float(np.mean(successful_counts)),
                    "graphs": graph_count,
                    "restarts": restarts,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["flow", "recurrent"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--route-width", type=int, default=3)
    parser.add_argument("--max-nodes", type=int, default=96)
    parser.add_argument("--train-depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--eval-graphs", type=int, default=64)
    parser.add_argument("--eval-restarts", type=int, default=32)
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
            name=f"multipath-transition-{args.model}-{os.environ.get('SLURM_JOB_ID', 'local')}",
            config=vars(args),
        )

    model = LocalGraphOperator(args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    generator = make_generator(device, args.seed + 17)
    history = []
    model.train()
    for step in range(1, args.steps + 1):
        adjacency, current, target = training_batch(
            args.batch_size,
            args.train_depth,
            args.route_width,
            args.max_nodes,
            device,
            generator,
        )
        if args.model == "flow":
            loss, metrics = flow_loss(model, adjacency, current, target, generator)
        else:
            loss, metrics = recurrent_loss(model, adjacency, current, target, generator)
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

    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "parameter_count": sum(p.numel() for p in model.parameters()),
        },
        output_dir / "checkpoint.pt",
    )
    with (output_dir / "train_history.json").open("w") as handle:
        json.dump(history, handle, indent=2)

    rows = evaluate(
        model,
        args.model,
        [4, 8, 16, 32],
        [1, 2, 4, 8],
        args.eval_graphs,
        args.eval_restarts,
        args.route_width,
        args.max_nodes,
        device,
        args.seed + 1000,
    )
    with (output_dir / "evaluation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "model": args.model,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "train_depth": args.train_depth,
        "evaluation": rows,
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
