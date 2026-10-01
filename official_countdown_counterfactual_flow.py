#!/usr/bin/env python
"""Causal flow response to a one-operator Countdown counterfactual."""

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
from countdown_task import OPS, evaluate_rpn
from modules.t5_encoder import get_encoder
from official_diagnostics import load_model
from official_flow_layer_causal_map import decode_state, prepare_flow
from official_graph_counterfactual_flow import switch_condition
from official_reasoning_velocity_transport import advance, finish, forward_velocity
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split
from utils.generation_utils import _dlm_decode_batch
from utils.sampling_utils import get_sampling_steps


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--depth", action="append", type=int, default=None)
    parser.add_argument("--pairs-per-depth", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--flow-steps", type=int, default=16)
    parser.add_argument("--time-index", action="append", type=int, default=None)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260928)
    return parser.parse_args()


def make_pair(row, tokenizer, rng):
    old_target = int(str(row["key"]).rsplit(":", 1)[1])
    tokens = str(row["target_text"]).split()
    edits = []
    for index, old_op in enumerate(tokens):
        if old_op not in OPS:
            continue
        for new_op in OPS:
            if new_op == old_op:
                continue
            changed = list(tokens)
            changed[index] = new_op
            expression = " ".join(changed)
            try:
                new_target = evaluate_rpn(expression, list(map(int, row["numbers"])))
            except ValueError:
                continue
            if new_target == old_target or not 0 < new_target <= 1000:
                continue
            prompt = str(row["input"]).replace(
                f"Target: {old_target}.", f"Target: {new_target}."
            )
            condition_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(expression, add_special_tokens=False)["input_ids"]
            if len(condition_ids) != len(row["condition_input_ids"]):
                continue
            if len(answer_ids) != len(row["input_ids"]):
                continue
            edits.append((index, old_op, new_op, new_target, prompt,
                          condition_ids, expression, answer_ids))
    if not edits:
        return None
    index, old_op, new_op, new_target, prompt, condition_ids, expression, answer_ids = rng.choice(edits)
    shared = {
        "operator_index": index,
        "old_operator": old_op,
        "new_operator": new_op,
        "source_numeric_target": old_target,
        "donor_numeric_target": new_target,
    }
    source = {**dict(row), **shared}
    donor = dict(row)
    donor.update(shared)
    donor.update({
        "condition_input_ids": condition_ids,
        "input_ids": answer_ids,
        "input": prompt,
        "target": expression,
        "target_text": expression,
        "answer": expression,
        "key": f"{tuple(row['numbers'])}:{new_target}",
        "example_id": f"{row['example_id']}-cf",
    })
    return source, donor


def build_pairs(dataset, tokenizer, depths, pairs_per_depth, seed):
    rng = random.Random(seed)
    sources, donors, counts = [], [], {}
    for depth in depths:
        candidates = [dict(row) for row in dataset if int(row["depth"]) == depth]
        rng.shuffle(candidates)
        start = len(sources)
        for row in candidates:
            pair = make_pair(row, tokenizer, rng)
            if pair is None:
                continue
            sources.append(pair[0])
            donors.append(pair[1])
            if len(sources) - start >= pairs_per_depth:
                break
        count = len(sources) - start
        if count < pairs_per_depth:
            raise RuntimeError(f"Only constructed {count}/{pairs_per_depth} pairs for d{depth}")
        counts[f"d{depth}"] = count
    return Dataset.from_list(sources), Dataset.from_list(donors), counts


def exact_answer_flags(state, model, config, source_ids, donor_ids):
    predicted = _dlm_decode_batch(state["z"], model, 1.0, config, 1.0)
    source_flags = torch.zeros(predicted.shape[0], dtype=torch.bool, device=predicted.device)
    donor_flags = torch.zeros_like(source_flags)
    for row, start in enumerate(state["starts"].tolist()):
        length = len(source_ids[row])
        span = predicted[row, start:start + length]
        source_flags[row] = span.eq(
            torch.tensor(source_ids[row], device=predicted.device)
        ).all()
        donor_flags[row] = span.eq(
            torch.tensor(donor_ids[row], device=predicted.device)
        ).all()
    return source_flags, donor_flags


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


def plot(metrics, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fields = [
        ("counterfactual_force_norm", "Counterfactual force norm"),
        ("one_step_counterfactual_progress", "One-step progress toward CF trajectory"),
        ("switched_to_counterfactual_rate", "Final counterfactual-answer rate"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    for depth in sorted({row["depth"] for row in metrics}):
        values = [row for row in metrics if row["depth"] == depth]
        for axis, (field, title) in zip(axes, fields):
            axis.plot([x["flow_time"] for x in values], [x[field] for x in values],
                      marker="o", label=f"countdown/d{depth}")
            axis.set_title(title); axis.set_xlabel("Flow intervention time"); axis.grid(alpha=.25)
    axes[2].set_ylim(-.03, 1.03); axes[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(output_dir / "countdown_counterfactual_response.png", dpi=220)
    plt.close(fig)


@torch.no_grad()
def run(args):
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config_from_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    model, checkpoint_step = load_model(config, args.checkpoint, encoder_config, tokenizer, device)
    validation = load_dataset_split(config.eval_data_path)
    depths = args.depth or [1, 2, 3, 4]
    sources, donors, counts = build_pairs(
        validation, tokenizer, depths, args.pairs_per_depth, args.seed
    )
    loader_args = dict(batch_size=args.batch_size, shuffle=False, num_workers=0,
        drop_last=False, max_seq_length=config.max_length,
        pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
        max_input_seq_length=config.max_input_length, distributed=False)
    source_loader = get_dataloader(sources, **loader_args)
    donor_loader = get_dataloader(donors, **loader_args)
    dtype = next(model.parameters()).dtype
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    t_steps = get_sampling_steps(args.flow_steps, "logit_normal", config.denoiser_p_mean,
                                 config.denoiser_p_std, device=device, dtype=dtype)
    time_indices = args.time_index or [0, 1, 2, 4, 8, 12, 15]
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    rows, cursor = [], 0
    for source_batch, donor_batch in zip(source_loader, donor_loader):
        bsz = len(source_batch["target"])
        noise = torch.randn((bsz, config.max_length, model.text_encoder_dim),
                            generator=generator, dtype=dtype)
        source = prepare_flow(source_batch, noise, model, encoder, tokenizer, config, device)
        donor = prepare_flow(donor_batch, noise, model, encoder, tokenizer, config, device)
        source_states, donor_states, source_velocities, donor_velocities = [], [], [], []
        for index in range(args.flow_steps):
            source_states.append(source); donor_states.append(donor)
            sv, sx = forward_velocity(model, source, t_steps[index], config, args.cfg, 1.)
            dv, dx = forward_velocity(model, donor, t_steps[index], config, args.cfg, 1.)
            source_velocities.append(sv); donor_velocities.append(dv)
            dt = t_steps[index + 1] - t_steps[index]
            source = advance(source, sv, sx, dt); donor = advance(donor, dv, dx, dt)
        source_stats = decode_state(source, model, config, 1.)
        donor_stats = decode_state(donor, model, config, 1.)
        source_ids = [
            list(map(int, sources[cursor + local]["input_ids"]))
            for local in range(bsz)
        ]
        donor_ids = [
            list(map(int, donors[cursor + local]["input_ids"]))
            for local in range(bsz)
        ]

        for time_index in time_indices:
            state = source_states[time_index]
            switched = switch_condition(state, donor_states[time_index])
            switched_v, _ = forward_velocity(model, switched, t_steps[time_index],
                                               config, args.cfg, 1.)
            dt = t_steps[time_index + 1] - t_steps[time_index]
            force_norm, progress = [], []
            for local, (start, length) in enumerate(zip(state["starts"], state["lengths"])):
                start, length = int(start), int(length)
                sl = slice(start, start + length)
                force = (switched_v[local, sl] - source_velocities[time_index][local, sl]).float()
                source_next = (state["z"] + dt * source_velocities[time_index])[local, sl].float()
                donor_state = donor_states[time_index]
                donor_next = (donor_state["z"] + dt * donor_velocities[time_index])[local, sl].float()
                switched_next = (switched["z"] + dt * switched_v)[local, sl].float()
                distance = (source_next - donor_next).norm().clamp_min(1e-8)
                force_norm.append(force.norm())
                progress.append((distance - (switched_next - donor_next).norm()) / distance)
            switched_final = finish(model, switched, time_index, t_steps, config, args.cfg, 1.)
            follows_source, follows_donor = exact_answer_flags(
                switched_final, model, config, source_ids, donor_ids
            )
            if time_index == 0 and not torch.equal(follows_donor, donor_stats["correct"]):
                raise RuntimeError("t=0 switch did not reproduce Countdown donor baseline")
            for local in range(bsz):
                meta = sources[cursor + local]
                rows.append({
                    "pair_index": cursor + local, "depth": int(meta["depth"]),
                    "operator_index": int(meta["operator_index"]),
                    "old_operator": meta["old_operator"], "new_operator": meta["new_operator"],
                    "time_index": time_index, "flow_time": float(t_steps[time_index]),
                    "force_norm": float(force_norm[local]),
                    "one_step_counterfactual_progress": float(progress[local]),
                    "source_baseline_correct": bool(source_stats["correct"][local]),
                    "donor_baseline_correct": bool(donor_stats["correct"][local]),
                    "eligible": bool(source_stats["correct"][local] and donor_stats["correct"][local]),
                    "switched_to_counterfactual": bool(follows_donor[local]),
                    "retained_original": bool(follows_source[local]),
                })
        cursor += bsz; print(f"Processed {cursor}/{len(sources)} pairs", flush=True)

    metrics = aggregate(rows)
    summary = {"checkpoint_step": checkpoint_step, "counts": counts,
        "flow_steps": args.flow_steps, "cfg": args.cfg, "time_indices": time_indices,
        "t_steps": [float(x) for x in t_steps], "metrics": metrics}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    with (output_dir / "per_sample.jsonl").open("w") as handle:
        for row in rows: handle.write(json.dumps(row) + "\n")
    plot(metrics, output_dir)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved Countdown counterfactual response to {output_dir}", flush=True)


if __name__ == "__main__":
    run(parse_args())
