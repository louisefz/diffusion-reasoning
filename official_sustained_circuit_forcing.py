#!/usr/bin/env python
"""Test whether sustained causal-circuit forcing creates a persistent mode."""

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
from official_causal_response_kernel import answer_logit_contrast, clone_state
from official_circuit_response_kernel import (
    capture_flow_heads, circuit_patched_velocity, parse_heads,
)
from official_counterfactual_patching import build_pairs_for_depth
from official_diagnostics import load_model
from official_flow_layer_causal_map import prepare_flow
from official_layerwise_patching import semantic_token_map
from official_reasoning_velocity_transport import advance, forward_velocity
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split
from utils.sampling_utils import get_sampling_steps


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--circuit-heads", type=parse_heads, required=True)
    parser.add_argument("--control-heads", type=parse_heads, required=True)
    parser.add_argument("--pairs-per-step", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--duration", action="append", type=int, default=None)
    parser.add_argument("--release-index", type=int, default=None,
                        help="If set, force the final `duration` steps before this fixed release state")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--self-cond-cfg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261001)
    return parser.parse_args()


def aggregate(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["mode"], row["duration"], row["release_index"],
                 row["observation_index"])].append(row)
    result = []
    fields = (
        "answer_axis_projection", "answer_response_norm",
        "logit_contrast_response", "donor_answer_rate", "source_answer_rate",
    )
    for (mode, duration, release_index, observation_index), items in sorted(buckets.items()):
        eligible = [item for item in items if item["eligible"]]
        chosen = eligible or items
        metric = {
            "mode": mode, "duration": duration,
            "release_index": release_index,
            "observation_index": observation_index,
            "observation_time": items[0]["observation_time"],
            "forcing_active": items[0]["forcing_active"],
            "samples": len(items), "eligible_samples": len(eligible),
        }
        for field in fields:
            metric[field] = float(np.mean([item[field] for item in chosen]))
        result.append(metric)
    return result


def final_summary(metrics, modes, durations):
    result = []
    for mode in modes:
        for duration in durations:
            chosen = sorted(
                [row for row in metrics if row["mode"] == mode
                 and row["duration"] == duration],
                key=lambda row: row["observation_index"])
            release_index = chosen[0]["release_index"]
            release = next(row for row in chosen
                           if row["observation_index"] == release_index)
            final = chosen[-1]
            release_projection = release["answer_axis_projection"]
            retention = (final["answer_axis_projection"] / release_projection
                         if abs(release_projection) > 1e-8 else 0.0)
            result.append({
                "mode": mode, "duration": duration,
                "release_index": release_index,
                "release_time": release["observation_time"],
                "release_projection": release_projection,
                "final_projection": final["answer_axis_projection"],
                "post_release_retention": retention,
                "final_logit_response": final["logit_contrast_response"],
                "final_donor_answer_rate": final["donor_answer_rate"],
                "final_source_answer_rate": final["source_answer_rate"],
                "eligible_samples": final["eligible_samples"],
            })
    return result


def plot(metrics, finals, output_dir, modes, durations, t_steps, depth):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(modes), 2, figsize=(12, 4.5 * len(modes)), squeeze=False)
    for row_index, mode in enumerate(modes):
        matrix = np.full((len(t_steps) - 1, len(durations)), np.nan)
        for col, duration in enumerate(durations):
            chosen = [row for row in metrics if row["mode"] == mode
                      and row["duration"] == duration]
            for row in chosen:
                matrix[row["observation_index"] - 1, col] = row["answer_axis_projection"]
        finite = matrix[np.isfinite(matrix)]
        vmax = max(float(np.quantile(np.abs(finite), .98)), 1e-6)
        image = axes[row_index, 0].imshow(
            matrix, origin="lower", aspect="auto", cmap="coolwarm",
            vmin=-vmax, vmax=vmax)
        axes[row_index, 0].set_xticks(range(len(durations)), durations)
        axes[row_index, 0].set_yticks(
            range(len(t_steps) - 1), [f"{float(t):.3g}" for t in t_steps[1:]])
        axes[row_index, 0].set_xlabel("Number of forced early steps")
        axes[row_index, 0].set_ylabel("Observation flow time")
        axes[row_index, 0].set_title(f"{mode}: donor-axis trajectory")
        fig.colorbar(image, ax=axes[row_index, 0], label="Answer-axis response")

        chosen_final = [row for row in finals if row["mode"] == mode]
        axes[row_index, 1].plot(
            durations, [row["final_projection"] for row in chosen_final],
            marker="o", label="endpoint response")
        axes[row_index, 1].plot(
            durations, [row["final_donor_answer_rate"] for row in chosen_final],
            marker="s", label="donor-answer rate")
        axes[row_index, 1].plot(
            durations, [row["post_release_retention"] for row in chosen_final],
            marker="^", label="post-release retention")
        axes[row_index, 1].set_xlabel("Number of forced early steps")
        axes[row_index, 1].set_title(f"{mode}: persistence after release")
        axes[row_index, 1].grid(alpha=.25)
        axes[row_index, 1].legend(frameon=False)
    fig.suptitle(f"d{depth}: sustained circuit forcing", y=.995)
    fig.tight_layout()
    fig.savefig(output_dir / "sustained_circuit_forcing.png", dpi=230)
    plt.close(fig)


@torch.no_grad()
def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    durations = sorted(set(args.duration or [1, 2, 3, 4, 5, 6]))
    if any(duration < 1 or duration >= args.flow_steps for duration in durations):
        raise ValueError("Durations must leave at least one unforced flow step")
    if args.release_index is not None:
        if not 1 <= args.release_index < args.flow_steps:
            raise ValueError("Fixed release index must leave at least one unforced step")
        if max(durations) > args.release_index:
            raise ValueError("Duration cannot exceed fixed release index")
    if len(args.circuit_heads) != len(args.control_heads):
        raise ValueError("Circuit and control groups must have equal size")
    modes = {
        "causal_circuit": args.circuit_heads,
        "size_matched_control": args.control_heads,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config_from_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    token_map = semantic_token_map(tokenizer)
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    model, checkpoint_step = load_model(config, args.checkpoint, encoder_config, tokenizer, device)
    validation = load_dataset_split(config.eval_data_path)
    originals, counterfactuals, counts = build_pairs_for_depth(
        validation, tokenizer, args.pairs_per_step, args.depth, args.seed)
    loader_args = dict(
        batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False,
        max_seq_length=config.max_length,
        pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
        max_input_seq_length=config.max_input_length, distributed=False)
    source_loader = get_dataloader(originals, **loader_args)
    donor_loader = get_dataloader(counterfactuals, **loader_args)
    dtype = next(model.parameters()).dtype
    t_steps = get_sampling_steps(
        args.flow_steps, "uniform", config.denoiser_p_mean,
        config.denoiser_p_std, device=device, dtype=dtype)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    use_bf16 = bool(config.use_bf16) and device.type == "cuda"
    rows = []
    cursor = 0

    for source_batch, donor_batch in zip(source_loader, donor_loader):
        bsz = len(source_batch["target"])
        noise = torch.randn((bsz, config.max_length, model.text_encoder_dim),
                            generator=generator, dtype=dtype)
        source = prepare_flow(source_batch, noise, model, encoder, tokenizer, config, device)
        donor = prepare_flow(donor_batch, noise, model, encoder, tokenizer, config, device)
        source_tokens = torch.tensor(
            [token_map[int(x)] for x in source_batch["target"]], device=device)
        donor_tokens = torch.tensor(
            [token_map[int(x)] for x in donor_batch["target"]], device=device)
        batch_rows = torch.arange(bsz, device=device)
        positions = source["positions"]

        source_states = [clone_state(source)]
        donor_states = [clone_state(donor)]
        donor_heads_by_time = []
        for index in range(args.flow_steps):
            donor_heads_by_time.append(capture_flow_heads(
                model, donor, t_steps[index], args.block,
                args.self_cond_cfg, use_bf16))
            source_v, source_x = forward_velocity(
                model, source, t_steps[index], config, args.cfg, args.self_cond_cfg)
            donor_v, donor_x = forward_velocity(
                model, donor, t_steps[index], config, args.cfg, args.self_cond_cfg)
            dt = t_steps[index + 1] - t_steps[index]
            source = advance(source, source_v, source_x, dt)
            donor = advance(donor, donor_v, donor_x, dt)
            source_states.append(clone_state(source))
            donor_states.append(clone_state(donor))

        source_contrasts = [answer_logit_contrast(
            state, model, config, args.self_cond_cfg, source_tokens, donor_tokens)[0]
            for state in source_states]
        source_final_prediction = answer_logit_contrast(
            source_states[-1], model, config, args.self_cond_cfg,
            source_tokens, donor_tokens)[1]
        donor_final_prediction = answer_logit_contrast(
            donor_states[-1], model, config, args.self_cond_cfg,
            source_tokens, donor_tokens)[1]
        eligible = ((source_final_prediction == source_tokens)
                    & (donor_final_prediction == donor_tokens))

        for mode, heads in modes.items():
            for duration in durations:
                branch = clone_state(source_states[0])
                release_index = args.release_index or duration
                forcing_start = release_index - duration
                for index in range(args.flow_steps):
                    forcing_active = forcing_start <= index < release_index
                    if forcing_active:
                        velocity, prediction = circuit_patched_velocity(
                            model, branch, donor_heads_by_time[index], heads,
                            args.strength, t_steps[index], config, args.cfg,
                            args.self_cond_cfg, args.block)
                    else:
                        velocity, prediction = forward_velocity(
                            model, branch, t_steps[index], config,
                            args.cfg, args.self_cond_cfg)
                    dt = t_steps[index + 1] - t_steps[index]
                    branch = advance(branch, velocity, prediction, dt)
                    observation_index = index + 1
                    source_obs = source_states[observation_index]
                    donor_obs = donor_states[observation_index]
                    response = (branch["z"][batch_rows, positions].float()
                                - source_obs["z"][batch_rows, positions].float())
                    donor_axis = (donor_obs["z"][batch_rows, positions].float()
                                  - source_obs["z"][batch_rows, positions].float())
                    axis_norm_sq = donor_axis.square().sum(dim=-1).clamp_min(1e-8)
                    projection = (response * donor_axis).sum(dim=-1) / axis_norm_sq
                    contrast, predicted = answer_logit_contrast(
                        branch, model, config, args.self_cond_cfg,
                        source_tokens, donor_tokens)
                    contrast_response = contrast - source_contrasts[observation_index]
                    for local in range(bsz):
                        rows.append({
                            "pair_index": cursor + local, "mode": mode,
                            "duration": duration,
                            "release_index": release_index,
                            "observation_index": observation_index,
                            "observation_time": float(t_steps[observation_index]),
                            "forcing_active": forcing_active,
                            "eligible": bool(eligible[local]),
                            "answer_axis_projection": float(projection[local]),
                            "answer_response_norm": float(response[local].norm()),
                            "logit_contrast_response": float(contrast_response[local]),
                            "donor_answer_rate": bool(predicted[local] == donor_tokens[local]),
                            "source_answer_rate": bool(predicted[local] == source_tokens[local]),
                        })

        cursor += bsz
        print(f"Sustained forcing d{args.depth}: processed {cursor}/{len(originals)}", flush=True)

    metrics = aggregate(rows)
    finals = final_summary(metrics, list(modes), durations)
    summary = {
        "checkpoint": args.checkpoint, "checkpoint_step": checkpoint_step,
        "depth": args.depth, "block": args.block,
        "circuit_heads": list(args.circuit_heads),
        "control_heads": list(args.control_heads),
        "counts": counts, "pairs": len(originals), "flow_steps": args.flow_steps,
        "t_steps": [float(x) for x in t_steps], "strength": args.strength,
        "durations": durations, "release_index": args.release_index,
        "metrics": metrics, "final_summary": finals,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "sustained_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader(); writer.writerows(metrics)
    with (output_dir / "final_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(finals[0]))
        writer.writeheader(); writer.writerows(finals)
    with (output_dir / "per_sample.jsonl").open("w") as handle:
        for row in rows: handle.write(json.dumps(row) + "\n")
    plot(metrics, finals, output_dir, list(modes), durations, t_steps, args.depth)
    print(f"Saved sustained forcing to {output_dir}", flush=True)


if __name__ == "__main__":
    run(parse_args())
