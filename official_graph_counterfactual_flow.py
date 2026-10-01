#!/usr/bin/env python
"""Measure the causal response of an ELF graph flow to one critical-edge edit."""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "official-elf" / "src"))

from configs.config import load_config_from_yaml
from graph_task import shortest_distances
from make_official_graph import render
from modules.t5_encoder import get_encoder
from official_diagnostics import load_model
from official_flow_layer_causal_map import decode_state, prepare_flow
from official_reasoning_velocity_transport import advance, finish, forward_velocity
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split
from utils.generation_utils import _dlm_decode_batch
from utils.sampling_utils import get_sampling_steps, restore_cond


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-data-path", default=None)
    parser.add_argument("--depth", action="append", type=int, default=None)
    parser.add_argument("--pairs-per-depth", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--flow-steps", type=int, default=16)
    parser.add_argument("--time-index", action="append", type=int, default=None)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260927)
    return parser.parse_args()


def critical_edges(row, num_nodes=16):
    edges = {tuple(edge) for edge in row["edges"]}
    result = []
    distances = shortest_distances(num_nodes, edges, int(row["source"]))
    for edge in sorted(edges):
        if shortest_distances(num_nodes, edges - {edge}, int(row["source"]))[
            int(row["query_target"])
        ] is None:
            result.append((edge, int(distances[edge[0]]) + 1))
    return result


def counterfactual_row(row, tokenizer, rng, num_nodes=16):
    """Replace one indispensable edge while preserving prompt token length."""
    original_edges = [tuple(edge) for edge in row["edges"]]
    edge_set = set(original_edges)
    candidates = critical_edges(row, num_nodes)
    rng.shuffle(candidates)
    for removed, edge_step in candidates:
        replacements = [
            (left, right)
            for left in range(num_nodes)
            for right in range(num_nodes)
            if left != right and (left, right) not in edge_set
        ]
        rng.shuffle(replacements)
        for replacement in replacements:
            edited = edge_set - {removed} | {replacement}
            if shortest_distances(num_nodes, edited, int(row["source"]))[
                int(row["query_target"])
            ] is not None:
                continue
            # Keep the edited edge in the removed edge's textual slot. This makes
            # the intervention local instead of shifting every later edge token.
            ordered = [replacement if edge == removed else edge for edge in original_edges]
            raw = {
                "edges": ordered,
                "task": "reach",
                "source": int(row["source"]),
                "target": int(row["query_target"]),
            }
            prompt = render(raw, num_nodes)
            ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            if len(ids) != len(row["condition_input_ids"]):
                continue
            donor = dict(row)
            donor.update({
                "condition_input_ids": ids,
                "input_ids": tokenizer("0", add_special_tokens=False)["input_ids"],
                "input": prompt,
                "target": "0",
                "edges": [list(edge) for edge in ordered],
                "states": [0] * (int(row["depth"]) + 1),
                "episode_id": f"{row['episode_id']}-cf",
                "graph_id": f"{row['graph_id']}-cf",
                "removed_edge": list(removed),
                "replacement_edge": list(replacement),
                "edge_step": edge_step,
            })
            source = dict(row)
            source.update({
                "removed_edge": list(removed),
                "replacement_edge": list(replacement),
                "edge_step": edge_step,
            })
            return source, donor
    return None


def build_pairs(dataset, tokenizer, depths, pairs_per_depth, seed):
    rng = random.Random(seed)
    sources, donors = [], []
    counts = {}
    for depth in depths:
        candidates = [
            dict(row) for row in dataset
            if row["task"] == "reach" and int(row["depth"]) == depth
            and str(row["target"]) == "1"
        ]
        rng.shuffle(candidates)
        for row in candidates:
            pair = counterfactual_row(row, tokenizer, rng)
            if pair is None:
                continue
            sources.append(pair[0])
            donors.append(pair[1])
            if sum(int(item["depth"]) == depth for item in sources) >= pairs_per_depth:
                break
        count = sum(int(item["depth"]) == depth for item in sources)
        if count < pairs_per_depth:
            raise RuntimeError(f"Only constructed {count}/{pairs_per_depth} pairs for d{depth}")
        counts[f"d{depth}"] = count
    return Dataset.from_list(sources), Dataset.from_list(donors), counts


def switch_condition(state, donor_state):
    result = dict(state)
    result["cond_seq"] = donor_state["cond_seq"]
    result["cond_mask"] = donor_state["cond_mask"]
    result["z"] = restore_cond(state["z"], result["cond_seq"], result["cond_mask"])
    result["previous"] = restore_cond(
        state["previous"], result["cond_seq"], result["cond_mask"]
    )
    return result


def aggregate(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["depth"], row["time_index"])].append(row)
    output = []
    for (depth, time_index), items in sorted(buckets.items()):
        eligible = [item for item in items if item["eligible"]]
        chosen = eligible or items
        output.append({
            "depth": depth,
            "time_index": time_index,
            "flow_time": items[0]["flow_time"],
            "samples": len(items),
            "eligible_samples": len(eligible),
            "source_accuracy": float(np.mean([x["source_baseline_correct"] for x in items])),
            "donor_accuracy": float(np.mean([x["donor_baseline_correct"] for x in items])),
            "counterfactual_force_norm": float(np.mean([x["force_norm"] for x in chosen])),
            "one_step_counterfactual_progress": float(np.mean([
                x["one_step_counterfactual_progress"] for x in chosen
            ])),
            "switched_to_counterfactual_rate": float(np.mean([
                x["switched_to_counterfactual"] for x in chosen
            ])),
            "retained_original_rate": float(np.mean([x["retained_original"] for x in chosen])),
        })
    return output


def make_plot(metrics, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    fields = [
        ("counterfactual_force_norm", "Counterfactual force norm"),
        ("one_step_counterfactual_progress", "One-step progress toward CF trajectory"),
        ("switched_to_counterfactual_rate", "Final counterfactual-answer rate"),
    ]
    for depth in sorted({row["depth"] for row in metrics}):
        selected = [row for row in metrics if row["depth"] == depth]
        for axis, (field, title) in zip(axes, fields):
            axis.plot([r["flow_time"] for r in selected], [r[field] for r in selected],
                      marker="o", label=f"reach/d{depth}")
            axis.set_title(title)
            axis.set_xlabel("Flow intervention time")
            axis.grid(alpha=0.25)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[2].set_ylim(-0.03, 1.03)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "graph_counterfactual_response.png", dpi=220)
    plt.close(fig)


def answer_flags(state, model, config, tokenizer):
    """Sequence-exact flags; T5 encodes '0' with two tokens but '1' with one."""
    predicted = _dlm_decode_batch(state["z"], model, 1.0, config, 1.0)
    zero_ids = tokenizer.encode("0", add_special_tokens=False)
    one_ids = tokenizer.encode("1", add_special_tokens=False)
    zero = torch.zeros(predicted.shape[0], dtype=torch.bool, device=predicted.device)
    one = torch.zeros_like(zero)
    for row, start in enumerate(state["starts"].tolist()):
        zero[row] = predicted[row, start:start + len(zero_ids)].eq(
            torch.tensor(zero_ids, device=predicted.device)
        ).all()
        one[row] = predicted[row, start:start + len(one_ids)].eq(
            torch.tensor(one_ids, device=predicted.device)
        ).all()
    return zero, one


@torch.no_grad()
def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config_from_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    model, checkpoint_step = load_model(
        config, args.checkpoint, encoder_config, tokenizer, device
    )
    validation = load_dataset_split(args.eval_data_path or config.eval_data_path)
    depths = args.depth or [4, 8, 12]
    sources, donors, counts = build_pairs(
        validation, tokenizer, depths, args.pairs_per_depth, args.seed
    )
    loader_args = dict(
        batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False,
        max_seq_length=config.max_length,
        pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
        max_input_seq_length=config.max_input_length, distributed=False,
    )
    source_loader = get_dataloader(sources, **loader_args)
    donor_loader = get_dataloader(donors, **loader_args)
    dtype = next(model.parameters()).dtype
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    t_steps = get_sampling_steps(
        args.flow_steps, "logit_normal", config.denoiser_p_mean,
        config.denoiser_p_std, device=device, dtype=dtype,
    )
    time_indices = args.time_index or [0, 1, 2, 4, 8, 12, 15]
    if any(index < 0 or index >= args.flow_steps for index in time_indices):
        raise ValueError("time-index must identify a velocity step")
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    rows = []
    cursor = 0
    for source_batch, donor_batch in zip(source_loader, donor_loader):
        bsz = len(source_batch["target"])
        noise = torch.randn(
            (bsz, config.max_length, model.text_encoder_dim),
            generator=generator, dtype=dtype,
        )
        source = prepare_flow(source_batch, noise, model, encoder, tokenizer, config, device)
        donor = prepare_flow(donor_batch, noise, model, encoder, tokenizer, config, device)
        source_states, donor_states = [], []
        source_velocities, donor_velocities = [], []
        for index in range(args.flow_steps):
            source_states.append(source)
            donor_states.append(donor)
            source_v, source_x = forward_velocity(
                model, source, t_steps[index], config, args.cfg, 1.0
            )
            donor_v, donor_x = forward_velocity(
                model, donor, t_steps[index], config, args.cfg, 1.0
            )
            source_velocities.append(source_v)
            donor_velocities.append(donor_v)
            dt = t_steps[index + 1] - t_steps[index]
            source = advance(source, source_v, source_x, dt)
            donor = advance(donor, donor_v, donor_x, dt)
        source_final, donor_final = source, donor
        source_stats = decode_state(source_final, model, config, 1.0)
        donor_stats = decode_state(donor_final, model, config, 1.0)
        batch_rows = torch.arange(bsz, device=device)
        starts = source_states[0]["starts"]
        width = max(
            len(tokenizer.encode("0", add_special_tokens=False)),
            len(tokenizer.encode("1", add_special_tokens=False)),
        )
        offsets = torch.arange(width, device=device)
        window = starts[:, None] + offsets[None, :]

        for time_index in time_indices:
            state = source_states[time_index]
            switched = switch_condition(state, donor_states[time_index])
            switched_v, _ = forward_velocity(
                model, switched, t_steps[time_index], config, args.cfg, 1.0
            )
            source_v = source_velocities[time_index]
            donor_v = donor_velocities[time_index]
            force = (
                switched_v[batch_rows[:, None], window].float()
                - source_v[batch_rows[:, None], window].float()
            ).flatten(1)
            dt = t_steps[time_index + 1] - t_steps[time_index]
            source_next = (
                state["z"] + dt * source_v
            )[batch_rows[:, None], window].float().flatten(1)
            donor_state = donor_states[time_index]
            donor_next = (
                donor_state["z"] + dt * donor_v
            )[batch_rows[:, None], window].float().flatten(1)
            switched_next = (
                switched["z"] + dt * switched_v
            )[batch_rows[:, None], window].float().flatten(1)
            native_distance = (source_next - donor_next).norm(dim=-1).clamp_min(1e-8)
            progress = (
                native_distance - (switched_next - donor_next).norm(dim=-1)
            ) / native_distance
            switched_final = finish(
                model, switched, time_index, t_steps, config, args.cfg, 1.0
            )
            follows_zero, follows_one = answer_flags(
                switched_final, model, config, tokenizer
            )
            if time_index == 0 and not torch.equal(
                follows_zero, donor_stats["correct"]
            ):
                raise RuntimeError(
                    "t=0 condition switch did not reproduce donor baseline"
                )
            for local in range(bsz):
                rows.append({
                    "pair_index": cursor + local,
                    "depth": int(sources[cursor + local]["depth"]),
                    "edge_step": int(sources[cursor + local]["edge_step"]),
                    "removed_edge": sources[cursor + local]["removed_edge"],
                    "replacement_edge": sources[cursor + local]["replacement_edge"],
                    "time_index": time_index,
                    "flow_time": float(t_steps[time_index]),
                    "force_norm": float(force[local].norm()),
                    "one_step_counterfactual_progress": float(progress[local]),
                    "source_baseline_correct": bool(source_stats["correct"][local]),
                    "donor_baseline_correct": bool(donor_stats["correct"][local]),
                    "eligible": bool(source_stats["correct"][local] and donor_stats["correct"][local]),
                    "switched_to_counterfactual": bool(follows_zero[local]),
                    "retained_original": bool(follows_one[local]),
                })
        cursor += bsz
        print(f"Processed {cursor}/{len(sources)} counterfactual pairs", flush=True)

    metrics = aggregate(rows)
    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": checkpoint_step,
        "seed": args.seed,
        "counts": counts,
        "flow_steps": args.flow_steps,
        "cfg": args.cfg,
        "time_indices": time_indices,
        "t_steps": [float(x) for x in t_steps],
        "metrics": metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    with (output_dir / "per_sample.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    make_plot(metrics, output_dir)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved graph counterfactual response to {output_dir}", flush=True)


if __name__ == "__main__":
    run(parse_args())
