#!/usr/bin/env python
"""Diagnose whether an ELF flow computes, transports, or corrects answers.

For each held-out input, run several deterministic ODE trajectories from
different Gaussian initial states.  The script measures two things:

1. Native decoding accuracy of the model's predicted clean endpoint x_pred at
   every flow time (including the first velocity evaluation).
2. Oracle wrong-to-correct rescue: for inputs with both a correct and an
   incorrect restart, transplant the correct restart's target state or its
   one-step target velocity into the incorrect restart and finish the flow.

The reverse correct-to-wrong intervention is included as a causal symmetry
control.  This is an oracle transport-capacity test, not a learned corrector.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "official-elf" / "src"))

from configs.config import load_config_from_yaml
from modules.t5_encoder import get_encoder
from official_diagnostics import load_model
from official_flow_layer_causal_map import decode_state, prepare_flow
from official_reasoning_velocity_transport import advance, finish, forward_velocity
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split
from utils.sampling_utils import get_sampling_steps


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--eval-data-path", default=None)
    p.add_argument("--difficulty", default=None,
                   help="Optionally keep only examples with this difficulty label.")
    p.add_argument("--num-samples", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--restarts", type=int, default=8)
    p.add_argument("--flow-steps", type=int, default=16)
    p.add_argument("--time-index", action="append", type=int, default=None)
    p.add_argument("--time-schedule", choices=("uniform", "logit_normal"), default="uniform")
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--self-cond-cfg", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20260928)
    return p.parse_args()


def repeat_batch(batch, repeats):
    keys = ("input_ids", "encoder_attention_mask", "cond_seq_mask", "target")
    return {key: np.repeat(np.asarray(batch[key]), repeats, axis=0) for key in keys}


def subset_batch(batch, indices):
    keys = ("input_ids", "encoder_attention_mask", "cond_seq_mask", "target")
    return {key: np.asarray(batch[key])[indices] for key in keys}


def clone_state(state):
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in state.items()}


def target_mask(state):
    length = state["z"].shape[1]
    positions = torch.arange(length, device=state["z"].device)[None, :]
    return ((positions >= state["starts"][:, None])
            & (positions < (state["starts"] + state["lengths"])[:, None]))


def transplant_state(source, donor):
    result = clone_state(source)
    mask = target_mask(source).unsqueeze(-1)
    result["z"] = torch.where(mask, donor["z"], source["z"])
    result["previous"] = torch.where(mask, donor["previous"], source["previous"])
    return result


def transplant_velocity(source, source_v, source_x, donor_v, donor_x, dt):
    mask = target_mask(source).unsqueeze(-1)
    velocity = torch.where(mask, donor_v, source_v)
    prediction = torch.where(mask, donor_x, source_x)
    return advance(source, velocity, prediction, dt)


def correctness(state, model, config, self_cond_cfg):
    return decode_state(state, model, config, self_cond_cfg)["correct"].bool()


def summarize_endpoint(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["time_index"]].append(row)
    result = []
    for time_index, items in sorted(buckets.items()):
        final_correct = [x for x in items if x["final_correct"]]
        final_wrong = [x for x in items if not x["final_correct"]]
        result.append({
            "time_index": time_index,
            "flow_time": items[0]["flow_time"],
            "trajectories": len(items),
            "predicted_endpoint_accuracy": float(np.mean([x["endpoint_correct"] for x in items])),
            "endpoint_accuracy_given_final_correct": float(np.mean(
                [x["endpoint_correct"] for x in final_correct])) if final_correct else None,
            "endpoint_accuracy_given_final_wrong": float(np.mean(
                [x["endpoint_correct"] for x in final_wrong])) if final_wrong else None,
            "final_accuracy": float(np.mean([x["final_correct"] for x in items])),
        })
    return result


def summarize_interventions(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["direction"], row["mode"], row["time_index"])].append(row)
    result = []
    for (direction, mode, time_index), items in sorted(buckets.items()):
        result.append({
            "direction": direction,
            "mode": mode,
            "time_index": time_index,
            "flow_time": items[0]["flow_time"],
            "pairs": len(items),
            "final_correct_rate": float(np.mean([x["final_correct"] for x in items])),
        })
    return result


def plot(endpoint, interventions, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axes[0].plot([x["flow_time"] for x in endpoint],
                 [x["predicted_endpoint_accuracy"] for x in endpoint], marker="o",
                 label="predicted endpoint")
    axes[0].axhline(endpoint[0]["final_accuracy"], color="black", linestyle="--",
                    label="final flow accuracy")
    axes[0].set(title="When is the answer natively readable?", xlabel="Flow time",
                ylabel="Exact accuracy", ylim=(-.03, 1.03))
    axes[0].grid(alpha=.25); axes[0].legend(frameon=False)

    styles = {"state_splice": "-", "velocity_pulse": "--"}
    for direction in ("wrong_to_correct", "correct_to_wrong"):
        for mode in ("state_splice", "velocity_pulse"):
            selected = [x for x in interventions
                        if x["direction"] == direction and x["mode"] == mode]
            if selected:
                axes[1].plot([x["flow_time"] for x in selected],
                             [x["final_correct_rate"] for x in selected], marker="o",
                             linestyle=styles[mode], label=f"{direction}/{mode}")
    axes[1].set(title="Can an oracle trajectory redirect the flow?", xlabel="Intervention time",
                ylabel="Final exact-correct rate", ylim=(-.03, 1.03))
    axes[1].grid(alpha=.25); axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(output_dir / "flow_error_correction.png", dpi=220)
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
    dataset = load_dataset_split(args.eval_data_path or config.eval_data_path)
    if args.difficulty is not None:
        if "difficulty" not in dataset.column_names:
            raise ValueError("--difficulty requested but the dataset has no difficulty column")
        dataset = dataset.filter(lambda row: str(row["difficulty"]) == args.difficulty)
    dataset = dataset.select(range(min(args.num_samples, len(dataset))))
    loader = get_dataloader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        drop_last=False, max_seq_length=config.max_length,
        pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
        max_input_seq_length=config.max_input_length, distributed=False,
    )
    dtype = next(model.parameters()).dtype
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    t_steps = get_sampling_steps(
        args.flow_steps, args.time_schedule, config.denoiser_p_mean,
        config.denoiser_p_std, device=device, dtype=dtype,
    )
    time_indices = args.time_index or [0, 1, 2, 4, 8, 12, 15]
    if any(i < 0 or i >= args.flow_steps for i in time_indices):
        raise ValueError("time-index must select a velocity evaluation")

    endpoint_rows, intervention_rows = [], []
    examples_seen = mixed_examples = 0
    for batch in loader:
        bsz = len(batch["target"])
        repeated = repeat_batch(batch, args.restarts)
        noises = torch.randn(
            (bsz, args.restarts, config.max_length, model.text_encoder_dim),
            generator=generator, dtype=dtype,
        )
        flat_noise = noises.reshape(
            bsz * args.restarts, config.max_length, model.text_encoder_dim)
        state = prepare_flow(repeated, flat_noise, model, encoder, tokenizer, config, device)
        endpoint_by_time = {}
        for index in range(args.flow_steps):
            velocity, prediction = forward_velocity(
                model, state, t_steps[index], config, args.cfg, args.self_cond_cfg)
            if index in time_indices:
                predicted_state = dict(state); predicted_state["z"] = prediction
                endpoint_by_time[index] = correctness(
                    predicted_state, model, config, args.self_cond_cfg).cpu().reshape(
                        bsz, args.restarts)
            state = advance(state, velocity, prediction, t_steps[index + 1] - t_steps[index])
        final = correctness(state, model, config, args.self_cond_cfg).cpu().reshape(
            bsz, args.restarts)
        for index in time_indices:
            endpoint_correct = endpoint_by_time[index]
            for local in range(bsz):
                for restart in range(args.restarts):
                    endpoint_rows.append({
                        "example_index": examples_seen + local,
                        "restart": restart,
                        "time_index": index,
                        "flow_time": float(t_steps[index]),
                        "endpoint_correct": bool(endpoint_correct[local, restart]),
                        "final_correct": bool(final[local, restart]),
                    })

        mixed = [i for i in range(bsz) if final[i].any() and (~final[i]).any()]
        if mixed:
            correct_restart = [int(torch.where(final[i])[0][0]) for i in mixed]
            wrong_restart = [int(torch.where(~final[i])[0][0]) for i in mixed]
            selected_batch = subset_batch(batch, mixed)
            correct_noise = torch.stack([noises[i, r] for i, r in zip(mixed, correct_restart)])
            wrong_noise = torch.stack([noises[i, r] for i, r in zip(mixed, wrong_restart)])
            correct_state = prepare_flow(
                selected_batch, correct_noise, model, encoder, tokenizer, config, device)
            wrong_state = prepare_flow(
                selected_batch, wrong_noise, model, encoder, tokenizer, config, device)
            correct_states, wrong_states = [], []
            correct_velocities, wrong_velocities = [], []
            correct_predictions, wrong_predictions = [], []
            for index in range(args.flow_steps):
                correct_states.append(correct_state); wrong_states.append(wrong_state)
                cv, cx = forward_velocity(
                    model, correct_state, t_steps[index], config, args.cfg, args.self_cond_cfg)
                wv, wx = forward_velocity(
                    model, wrong_state, t_steps[index], config, args.cfg, args.self_cond_cfg)
                correct_velocities.append(cv); wrong_velocities.append(wv)
                correct_predictions.append(cx); wrong_predictions.append(wx)
                dt = t_steps[index + 1] - t_steps[index]
                correct_state = advance(correct_state, cv, cx, dt)
                wrong_state = advance(wrong_state, wv, wx, dt)

            for index in time_indices:
                dt = t_steps[index + 1] - t_steps[index]
                directions = (
                    ("wrong_to_correct", wrong_states, correct_states,
                     wrong_velocities, correct_velocities,
                     wrong_predictions, correct_predictions),
                    ("correct_to_wrong", correct_states, wrong_states,
                     correct_velocities, wrong_velocities,
                     correct_predictions, wrong_predictions),
                )
                for direction, source_states, donor_states, source_vs, donor_vs, source_xs, donor_xs in directions:
                    spliced = transplant_state(source_states[index], donor_states[index])
                    spliced_final = finish(
                        model, spliced, index, t_steps, config,
                        args.cfg, args.self_cond_cfg)
                    spliced_correct = correctness(
                        spliced_final, model, config, args.self_cond_cfg).cpu()
                    pulsed = transplant_velocity(
                        source_states[index], source_vs[index], source_xs[index],
                        donor_vs[index], donor_xs[index], dt)
                    pulsed_final = finish(
                        model, pulsed, index + 1, t_steps, config,
                        args.cfg, args.self_cond_cfg)
                    pulsed_correct = correctness(
                        pulsed_final, model, config, args.self_cond_cfg).cpu()
                    for local in range(len(mixed)):
                        base = {
                            "example_index": examples_seen + mixed[local],
                            "direction": direction,
                            "time_index": index,
                            "flow_time": float(t_steps[index]),
                        }
                        intervention_rows.append({
                            **base, "mode": "state_splice",
                            "final_correct": bool(spliced_correct[local]),
                        })
                        intervention_rows.append({
                            **base, "mode": "velocity_pulse",
                            "final_correct": bool(pulsed_correct[local]),
                        })
            mixed_examples += len(mixed)
        examples_seen += bsz
        print(f"Processed {examples_seen}/{len(dataset)}; mixed={mixed_examples}", flush=True)

    endpoint = summarize_endpoint(endpoint_rows)
    interventions = summarize_interventions(intervention_rows)
    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": checkpoint_step,
        "samples": examples_seen,
        "restarts": args.restarts,
        "mixed_examples": mixed_examples,
        "mixed_example_rate": mixed_examples / max(examples_seen, 1),
        "flow_steps": args.flow_steps,
        "time_schedule": args.time_schedule,
        "cfg": args.cfg,
        "difficulty": args.difficulty,
        "t_steps": [float(x) for x in t_steps],
        "endpoint_readout": endpoint,
        "interventions": interventions,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for filename, rows in (("endpoint_readout.csv", endpoint),
                           ("interventions.csv", interventions)):
        if rows:
            with (output_dir / filename).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
    with (output_dir / "per_trajectory.jsonl").open("w") as handle:
        for row in endpoint_rows:
            handle.write(json.dumps({"kind": "endpoint", **row}) + "\n")
        for row in intervention_rows:
            handle.write(json.dumps({"kind": "intervention", **row}) + "\n")
    plot(endpoint, interventions, output_dir)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved flow correction diagnostic to {output_dir}", flush=True)


if __name__ == "__main__":
    run(parse_args())
