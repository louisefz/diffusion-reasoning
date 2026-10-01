#!/usr/bin/env python
"""Track native wrong/correct transitions along an ELF reasoning flow.

Unlike donor interventions, this diagnostic never changes the trajectory.  It
decodes every predicted clean endpoint and the final integrated state, then
counts exact-answer and target-token W->C / C->W transitions.  This separates
autonomous correction from mere oracle controllability.
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
from official_flow_layer_causal_map import prepare_flow
from official_flow_error_correction import repeat_batch
from official_reasoning_velocity_transport import advance, forward_velocity
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split
from utils.generation_utils import _dlm_decode_batch
from utils.sampling_utils import get_sampling_steps


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


def decode_correctness(state, model, config, self_cond_cfg):
    ids = _dlm_decode_batch(state["z"], model, 1.0, config, self_cond_cfg)
    positions = torch.arange(ids.shape[1], device=ids.device)[None, :]
    mask = ((positions >= state["starts"][:, None])
            & (positions < (state["starts"] + state["lengths"])[:, None]))
    token_correct = ids.eq(state["input_ids"])
    exact = (token_correct | ~mask).all(dim=1)
    return token_correct.cpu(), mask.cpu(), exact.cpu()


def group_names(dataset, start, size, restarts):
    rows = dataset[start:start + size]
    if "difficulty" in dataset.column_names:
        base = [str(x) for x in rows["difficulty"]]
    elif "task" in dataset.column_names and "depth" in dataset.column_names:
        base = [f"{task}/d{depth}" for task, depth in zip(rows["task"], rows["depth"])]
    else:
        base = ["all"] * size
    return [group for group in base for _ in range(restarts)]


def add_state(acc, group, state_index, flow_time, token_correct, mask, exact):
    key = (group, state_index, flow_time)
    row = acc[key]
    row["trajectories"] += 1
    row["exact_correct"] += int(exact)
    row["token_correct"] += int((token_correct & mask).sum())
    row["target_tokens"] += int(mask.sum())


def add_transition(acc, group, left_index, right_index, left_time, right_time,
                   left_token, right_token, mask, left_exact, right_exact):
    key = (group, left_index, right_index, left_time, right_time)
    row = acc[key]
    row["trajectories"] += 1
    row["exact_wc"] += int((not left_exact) and right_exact)
    row["exact_cw"] += int(left_exact and (not right_exact))
    row["exact_ww"] += int((not left_exact) and (not right_exact))
    row["exact_cc"] += int(left_exact and right_exact)
    row["token_wc"] += int(((~left_token) & right_token & mask).sum())
    row["token_cw"] += int((left_token & (~right_token) & mask).sum())
    row["token_ww"] += int(((~left_token) & (~right_token) & mask).sum())
    row["token_cc"] += int((left_token & right_token & mask).sum())


def materialize_states(acc):
    rows = []
    for (group, index, time), x in sorted(acc.items()):
        rows.append({
            "group": group, "state_index": index, "flow_time": time,
            **x,
            "exact_accuracy": x["exact_correct"] / max(x["trajectories"], 1),
            "target_token_accuracy": x["token_correct"] / max(x["target_tokens"], 1),
        })
    return rows


def materialize_transitions(acc):
    rows = []
    for (group, li, ri, lt, rt), x in sorted(acc.items()):
        exact_wrong = x["exact_wc"] + x["exact_ww"]
        exact_correct = x["exact_cw"] + x["exact_cc"]
        token_wrong = x["token_wc"] + x["token_ww"]
        token_correct = x["token_cw"] + x["token_cc"]
        rows.append({
            "group": group, "left_index": li, "right_index": ri,
            "left_time": lt, "right_time": rt, **x,
            "exact_wrong_to_correct_rate": x["exact_wc"] / max(exact_wrong, 1),
            "exact_correct_to_wrong_rate": x["exact_cw"] / max(exact_correct, 1),
            "exact_net_correction_rate": (x["exact_wc"] - x["exact_cw"]) / max(x["trajectories"], 1),
            "token_wrong_to_correct_rate": x["token_wc"] / max(token_wrong, 1),
            "token_correct_to_wrong_rate": x["token_cw"] / max(token_correct, 1),
            "token_net_correction_rate": (x["token_wc"] - x["token_cw"]) / max(
                x["token_wc"] + x["token_cw"] + x["token_ww"] + x["token_cc"], 1),
        })
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def make_plot(states, transitions, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = sorted({x["group"] for x in states})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex="col")
    for group in groups:
        s = [x for x in states if x["group"] == group]
        tr = [x for x in transitions if x["group"] == group and x["right_index"] == x["left_index"] + 1]
        axes[0, 0].plot([x["flow_time"] for x in s], [x["exact_accuracy"] for x in s], marker="o", label=group)
        axes[0, 1].plot([x["flow_time"] for x in s], [x["target_token_accuracy"] for x in s], marker="o", label=group)
        axes[1, 0].plot([x["right_time"] for x in tr], [x["exact_net_correction_rate"] for x in tr], marker="o", label=group)
        axes[1, 1].plot([x["right_time"] for x in tr], [x["token_net_correction_rate"] for x in tr], marker="o", label=group)
    titles = ("Exact answer accuracy", "Target-token accuracy",
              "Exact net autonomous correction", "Token net autonomous correction")
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title); ax.grid(alpha=.25); ax.axhline(0, color="black", lw=.7)
    axes[0, 0].legend(fontsize=8); axes[1, 0].set_xlabel("Flow time")
    axes[1, 1].set_xlabel("Flow time")
    fig.tight_layout(); fig.savefig(output_dir / "autonomous_correction.png", dpi=220)
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
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        drop_last=False, max_seq_length=config.max_length,
        pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
        max_input_seq_length=config.max_input_length, distributed=False,
    )
    dtype = next(model.parameters()).dtype
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    t_steps = get_sampling_steps(
        args.flow_steps, args.time_schedule, config.denoiser_p_mean,
        config.denoiser_p_std, device=device, dtype=dtype,
    )
    state_acc = defaultdict(lambda: defaultdict(int))
    transition_acc = defaultdict(lambda: defaultdict(int))
    seen = 0
    for batch in loader:
        bsz = len(batch["target"])
        repeated = repeat_batch(batch, args.restarts)
        noise = torch.randn(
            (bsz * args.restarts, config.max_length, model.text_encoder_dim),
            generator=generator, dtype=dtype,
        )
        state = prepare_flow(repeated, noise, model, encoder, tokenizer, config, device)
        snapshots = []
        for index in range(args.flow_steps):
            velocity, prediction = forward_velocity(
                model, state, t_steps[index], config, args.cfg, args.self_cond_cfg)
            predicted_state = dict(state); predicted_state["z"] = prediction
            snapshots.append((*decode_correctness(
                predicted_state, model, config, args.self_cond_cfg), float(t_steps[index])))
            state = advance(state, velocity, prediction, t_steps[index + 1] - t_steps[index])
        snapshots.append((*decode_correctness(
            state, model, config, args.self_cond_cfg), 1.0))
        groups = group_names(dataset, seen, bsz, args.restarts)
        for row, group in enumerate(groups):
            for index, (correct, mask, exact, time) in enumerate(snapshots):
                add_state(state_acc, group, index, time, correct[row], mask[row], bool(exact[row]))
            for index in range(len(snapshots) - 1):
                lc, lm, le, lt = snapshots[index]
                rc, _, re, rt = snapshots[index + 1]
                add_transition(transition_acc, group, index, index + 1, lt, rt,
                               lc[row], rc[row], lm[row], bool(le[row]), bool(re[row]))
            # Each intermediate prediction directly against the final integrated state.
            fc, _, fe, ft = snapshots[-1]
            for index in range(len(snapshots) - 1):
                lc, lm, le, lt = snapshots[index]
                add_transition(transition_acc, group, index, len(snapshots) - 1, lt, ft,
                               lc[row], fc[row], lm[row], bool(le[row]), bool(fe[row]))
        seen += bsz
        print(f"Processed {seen}/{len(dataset)}", flush=True)

    states = materialize_states(state_acc)
    transitions = materialize_transitions(transition_acc)
    summary = {
        "checkpoint": args.checkpoint, "checkpoint_step": checkpoint_step,
        "samples": seen, "restarts": args.restarts, "cfg": args.cfg,
        "self_cond_cfg": args.self_cond_cfg, "flow_steps": args.flow_steps,
        "time_schedule": args.time_schedule, "difficulty": args.difficulty,
        "states": states, "transitions": transitions,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(output_dir / "states.csv", states)
    write_csv(output_dir / "transitions.csv", transitions)
    make_plot(states, transitions, output_dir)
    print(f"Saved autonomous-correction diagnostic to {output_dir}", flush=True)


if __name__ == "__main__":
    main(parse_args())
