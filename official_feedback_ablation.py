#!/usr/bin/env python
"""Causally test whether ELF's iterative self-conditioning drives correction.

All conditions share inputs and initial noise.  We change only the clean-endpoint
prediction fed back to the next velocity evaluation:

  normal  : native previous prediction
  zero    : remove target-side feedback (an in-distribution training condition)
  frozen  : reuse the first prediction at every later step
  shuffled: exchange feedback among restarts of the same problem

The last two controls distinguish useful iterative updating from merely giving
the network any endpoint-shaped vector.
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
from official_autonomous_correction import decode_correctness, group_names
from official_diagnostics import load_model
from official_flow_error_correction import repeat_batch
from official_flow_layer_causal_map import prepare_flow
from official_reasoning_velocity_transport import advance, forward_velocity
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split
from utils.sampling_utils import get_sampling_steps, restore_cond


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--eval-data-path", default=None)
    p.add_argument("--difficulty", default=None)
    p.add_argument("--num-samples", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--flow-steps", type=int, default=16)
    p.add_argument("--time-schedule", choices=("uniform", "logit_normal"), default="uniform")
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--self-cond-cfg", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20260928)
    return p.parse_args()


def feedback_for(mode, state, first_prediction, bsz, restarts):
    if mode == "normal":
        return state["previous"]
    if mode == "zero":
        return restore_cond(torch.zeros_like(state["previous"]), state["cond_seq"], state["cond_mask"])
    if mode == "frozen":
        if first_prediction is None:
            return state["previous"]
        return first_prediction
    if mode == "shuffled":
        if first_prediction is None:
            return state["previous"]
        shaped = state["previous"].reshape(bsz, restarts, *state["previous"].shape[1:])
        # Deterministic cyclic exchange; every trajectory receives another
        # restart's state while prompt/condition positions are restored below.
        shifted = torch.roll(shaped, shifts=1, dims=1).reshape_as(state["previous"])
        return restore_cond(shifted, state["cond_seq"], state["cond_mask"])
    raise ValueError(mode)


def add(acc, mode, group, index, time, correct, mask, exact):
    row = acc[(mode, group, index, time)]
    row["trajectories"] += 1
    row["exact_correct"] += int(exact)
    row["token_correct"] += int((correct & mask).sum())
    row["target_tokens"] += int(mask.sum())


def materialize(acc):
    rows = []
    for (mode, group, index, time), x in sorted(acc.items()):
        rows.append({
            "mode": mode, "group": group, "state_index": index, "flow_time": time,
            **x,
            "exact_accuracy": x["exact_correct"] / max(x["trajectories"], 1),
            "target_token_accuracy": x["token_correct"] / max(x["target_tokens"], 1),
        })
    return rows


def plot(rows, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = sorted({x["group"] for x in rows})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for mode in ("normal", "zero", "frozen", "shuffled"):
        chosen = [x for x in rows if x["mode"] == mode]
        # Pool counts rather than average group accuracies.
        by_index = defaultdict(lambda: defaultdict(float))
        for x in chosen:
            r = by_index[x["state_index"]]
            r["time"] = x["flow_time"]
            r["n"] += x["trajectories"]
            r["correct"] += x["exact_correct"]
            r["tokens"] += x["target_tokens"]
            r["token_correct"] += x["token_correct"]
        xs = sorted(by_index)
        axes[0].plot([by_index[i]["time"] for i in xs],
                     [by_index[i]["correct"] / by_index[i]["n"] for i in xs],
                     marker="o", label=mode)
        axes[1].plot([by_index[i]["time"] for i in xs],
                     [by_index[i]["token_correct"] / by_index[i]["tokens"] for i in xs],
                     marker="o", label=mode)
    axes[0].set(title="Exact answer formation", ylabel="Exact accuracy", xlabel="Flow time")
    axes[1].set(title="Target-token formation", ylabel="Token accuracy", xlabel="Flow time")
    for ax in axes:
        ax.grid(alpha=.25); ax.legend(frameon=False)
    fig.suptitle("Self-conditioning feedback ablation: " + ", ".join(groups), fontsize=10)
    fig.tight_layout(); fig.savefig(output_dir / "feedback_ablation.png", dpi=220)
    plt.close(fig)


@torch.no_grad()
def main(args):
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config_from_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    model, checkpoint_step = load_model(config, args.checkpoint, encoder_config, tokenizer, device)
    dataset = load_dataset_split(args.eval_data_path or config.eval_data_path)
    if args.difficulty is not None:
        dataset = dataset.filter(lambda row: str(row["difficulty"]) == args.difficulty)
    dataset = dataset.select(range(min(args.num_samples, len(dataset))))
    loader = get_dataloader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False,
        max_seq_length=config.max_length, pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
        max_input_seq_length=config.max_input_length, distributed=False,
    )
    dtype = next(model.parameters()).dtype
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    t_steps = get_sampling_steps(args.flow_steps, args.time_schedule,
                                 config.denoiser_p_mean, config.denoiser_p_std,
                                 device=device, dtype=dtype)
    modes = ("normal", "zero", "frozen", "shuffled")
    acc = defaultdict(lambda: defaultdict(int)); trajectory_rows = []; seen = 0
    for batch in loader:
        bsz = len(batch["target"]); repeated = repeat_batch(batch, args.restarts)
        noise = torch.randn((bsz * args.restarts, config.max_length, model.text_encoder_dim),
                            generator=generator, dtype=dtype)
        initial = prepare_flow(repeated, noise, model, encoder, tokenizer, config, device)
        groups = group_names(dataset, seen, bsz, args.restarts)
        for mode in modes:
            state = {k: v.clone() if torch.is_tensor(v) else v for k, v in initial.items()}
            first_prediction = None
            for index in range(args.flow_steps):
                model_state = dict(state)
                model_state["previous"] = feedback_for(
                    mode, state, first_prediction, bsz, args.restarts)
                velocity, prediction = forward_velocity(
                    model, model_state, t_steps[index], config, args.cfg, args.self_cond_cfg)
                if first_prediction is None:
                    first_prediction = prediction.clone()
                predicted_state = dict(state); predicted_state["z"] = prediction
                correct, mask, exact = decode_correctness(
                    predicted_state, model, config, args.self_cond_cfg)
                for row, group in enumerate(groups):
                    add(acc, mode, group, index, float(t_steps[index]),
                        correct[row], mask[row], bool(exact[row]))
                # Store the native current prediction.  The intervention is
                # applied only at the next model call by feedback_for().
                state = advance(state, velocity, prediction, t_steps[index + 1] - t_steps[index])
            correct, mask, exact = decode_correctness(state, model, config, args.self_cond_cfg)
            for row, group in enumerate(groups):
                add(acc, mode, group, args.flow_steps, 1.0,
                    correct[row], mask[row], bool(exact[row]))
                trajectory_rows.append({
                    "mode": mode,
                    "example_index": seen + row // args.restarts,
                    "restart": row % args.restarts,
                    "group": group,
                    "exact_correct": int(bool(exact[row])),
                    "target_token_accuracy": float(
                        (correct[row] & mask[row]).sum().item() / max(mask[row].sum().item(), 1)),
                })
        seen += bsz; print(f"Processed {seen}/{len(dataset)}", flush=True)

    rows = materialize(acc)
    summary = {"checkpoint": args.checkpoint, "checkpoint_step": checkpoint_step,
               "samples": seen, "restarts": args.restarts, "cfg": args.cfg,
               "self_cond_cfg": args.self_cond_cfg, "flow_steps": args.flow_steps,
               "difficulty": args.difficulty, "modes": list(modes), "rows": rows}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "feedback_ablation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (output_dir / "trajectories.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trajectory_rows[0]))
        writer.writeheader(); writer.writerows(trajectory_rows)
    plot(rows, output_dir)
    print(f"Saved feedback ablation to {output_dir}", flush=True)


if __name__ == "__main__":
    main(parse_args())
