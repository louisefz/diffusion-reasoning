#!/usr/bin/env python
"""Measure causal answer stability by perturbing and branching ELF flow states."""

import argparse
from collections import defaultdict
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent
OFFICIAL_SRC = ROOT / "official-elf" / "src"
sys.path.insert(0, str(OFFICIAL_SRC))

from configs.config import load_config_from_yaml
from modules.t5_encoder import get_encoder
from official_diagnostics import load_model
from official_flow_trajectory import (
    DEFAULT_GROUPS, answer_decoder_stats, parse_group_spec,
    select_balanced_indices,
)
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split
from utils.encoder_utils import encode_text
from utils.generation_utils import _dlm_decode_logits
from utils.sampling_utils import _ode_step, get_sampling_steps, restore_cond


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group", action="append", default=None,
                        help="TASK:DEPTH:COUNT; repeat for balanced groups")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--schedule", choices=("uniform", "logit_normal"),
                        default="logit_normal")
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--self-cond-cfg", type=float, default=1.0)
    parser.add_argument("--branch-index", action="append", type=int, default=None)
    parser.add_argument("--sigma", action="append", type=float, default=None,
                        help="Per-coordinate perturbation SD as a fraction of z RMS")
    parser.add_argument(
        "--perturb-target", action="append",
        choices=("z", "selfcond", "both"), default=None,
        help="Which component of the Markov sampling state to perturb",
    )
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260918)
    return parser.parse_args()


def _answer_agreement(predictions, references, starts, lengths):
    result = torch.ones(predictions.shape[0], dtype=torch.bool,
                        device=predictions.device)
    for row in range(predictions.shape[0]):
        start = int(starts[row])
        stop = start + int(lengths[row])
        result[row] = predictions[row, start:stop].eq(
            references[row, start:stop]
        ).all()
    return result


def _generated_rms(z, cond_mask):
    generated = (1.0 - cond_mask).unsqueeze(-1).float()
    denom = generated.sum(dim=(1, 2)) * z.shape[-1]
    energy = (z.float().square() * generated).sum(dim=(1, 2))
    return torch.sqrt(energy / denom.clamp_min(1.0))


def _mean_se(values):
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    se = float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0
    return mean, se


def aggregate_rows(per_branch_rows):
    by_condition_sample = defaultdict(list)
    for row in per_branch_rows:
        key = (row["group"], row["perturb_target"], row["t_index"],
               row["t_state"], row["sigma"], row["source_id"],
               row["baseline_correct"])
        by_condition_sample[key].append(row)

    by_condition = defaultdict(list)
    for key, rows in by_condition_sample.items():
        (group, perturb_target, t_index, t_state, sigma, source_id,
         baseline_correct) = key
        by_condition[(group, perturb_target, t_index, t_state, sigma)].append({
            "source_id": source_id,
            "baseline_correct": baseline_correct,
            "correct": np.mean([row["correct"] for row in rows]),
            "agreement": np.mean([row["agrees_with_baseline"] for row in rows]),
            "z_noise_rms": np.mean([row["z_noise_rms"] for row in rows]),
            "selfcond_noise_rms": np.mean(
                [row["selfcond_noise_rms"] for row in rows]
            ),
            "rollouts": len(rows),
        })

    aggregate = []
    for (group, perturb_target, t_index, t_state, sigma), samples in sorted(
        by_condition.items()
    ):
        correct, correct_se = _mean_se([row["correct"] for row in samples])
        agreement, agreement_se = _mean_se([row["agreement"] for row in samples])
        retained = [row["correct"] for row in samples if row["baseline_correct"]]
        retention, retention_se = _mean_se(retained) if retained else (None, None)
        aggregate.append({
            "group": group,
            "perturb_target": perturb_target,
            "t_index": t_index,
            "t_state": t_state,
            "sigma": sigma,
            "samples": len(samples),
            "rollouts": sum(row["rollouts"] for row in samples),
            "baseline_accuracy": np.mean(
                [row["baseline_correct"] for row in samples]
            ),
            "correct_rate": correct,
            "correct_se_across_examples": correct_se,
            "agreement_rate": agreement,
            "agreement_se_across_examples": agreement_se,
            "retention_rate_given_baseline_correct": retention,
            "retention_se_across_examples": retention_se,
            "mean_z_noise_rms": float(np.mean(
                [row["z_noise_rms"] for row in samples]
            )),
            "mean_selfcond_noise_rms": float(np.mean(
                [row["selfcond_noise_rms"] for row in samples]
            )),
        })
    return aggregate


def _save_csv(rows, path):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows, output_dir, metric, filename, ylabel, perturb_target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [row for row in rows if row["perturb_target"] == perturb_target]
    groups = sorted({row["group"] for row in rows})
    sigmas = sorted({row["sigma"] for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for axis, group in zip(axes.flat, groups):
        for sigma in sigmas:
            selected = [row for row in rows
                        if row["group"] == group and row["sigma"] == sigma]
            axis.plot(
                [row["t_state"] for row in selected],
                [row[metric] for row in selected],
                marker="o", label=f"sigma={sigma:g}",
            )
        axis.axhline(0.9, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        axis.set_title(f"{group} — {perturb_target}")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("Flow time")
    for axis in axes[:, 0]:
        axis.set_ylabel(ylabel)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=180)
    plt.close(fig)


@torch.no_grad()
def run(args):
    if args.branches < 1:
        raise ValueError("--branches must be positive")
    sigmas = sorted(set(args.sigma or [0.0, 0.01, 0.03, 0.1, 0.3]))
    if not sigmas or sigmas[0] < 0:
        raise ValueError("sigmas must be non-negative")
    branch_indices = sorted(set(args.branch_index or [2, 7, 13, 14]))
    perturb_targets = list(dict.fromkeys(
        args.perturb_target or ["z", "selfcond", "both"]
    ))
    if any(index <= 0 or index >= args.steps for index in branch_indices):
        raise ValueError("branch indices must be between 1 and steps-1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config_from_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name or config.encoder_model_name
    )
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    dataset = load_dataset_split(config.eval_data_path)
    specs = args.group or list(DEFAULT_GROUPS)
    indices, metadata = select_balanced_indices(dataset, specs, args.seed)
    selected_dataset = dataset.select(indices)
    model, checkpoint_step = load_model(
        config, args.checkpoint, encoder_config, tokenizer, device,
    )
    dtype = next(model.parameters()).dtype

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    t_steps = get_sampling_steps(
        n_steps=args.steps, time_schedule=args.schedule,
        P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
        device=device, dtype=dtype,
    )
    base_noise_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    branch_generator = torch.Generator(device=device).manual_seed(args.seed + 991)
    loader = get_dataloader(
        selected_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, drop_last=False, max_seq_length=config.max_length,
        pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
        max_input_seq_length=config.max_input_length, distributed=False,
    )

    per_branch_rows = []
    cursor = 0
    use_bf16 = bool(config.use_bf16) and device.type == "cuda"
    replicas = len(sigmas) * args.branches
    sigma_pattern = torch.tensor(sigmas, device=device, dtype=torch.float32).repeat_interleave(
        args.branches
    )

    for batch in loader:
        bsz = len(batch["target"])
        batch_meta = metadata[cursor:cursor + bsz]
        cursor += bsz
        input_ids = torch.from_numpy(np.asarray(batch["input_ids"])).to(device).long()
        encoder_mask = torch.from_numpy(
            np.asarray(batch["encoder_attention_mask"])
        ).to(device).float()
        cond_mask = torch.from_numpy(np.asarray(batch["cond_seq_mask"])).to(device).float()
        answer_starts = cond_mask.to(torch.long).sum(dim=1)
        answer_lengths = torch.tensor(
            [len(tokenizer.encode(str(target), add_special_tokens=False))
             for target in batch["target"]], device=device, dtype=torch.long,
        )
        cond_seq = encode_text(
            input_ids, encoder_mask, encoder, config.latent_mean,
            config.latent_std, use_bf16=bool(config.use_bf16),
        ).to(dtype)
        z = (torch.randn(
            (bsz, config.max_length, model.text_encoder_dim),
            generator=base_noise_generator, dtype=dtype,
        ) * config.denoiser_noise_scale).to(device)
        z = restore_cond(z, cond_seq, cond_mask)
        xpred = restore_cond(torch.zeros_like(z), cond_seq, cond_mask)
        branch_states = {}

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            for step_index in range(len(t_steps) - 1):
                z, xpred = _ode_step(
                    model=model, z=z, t=float(t_steps[step_index]),
                    t_next=float(t_steps[step_index + 1]), x_pred_prev=xpred,
                    config=config, cfg_scale=args.cfg,
                    self_cond_cfg_scale=args.self_cond_cfg,
                    cond_seq=cond_seq, cond_seq_mask=cond_mask,
                )
                state_index = step_index + 1
                if state_index in branch_indices:
                    branch_states[state_index] = (z.clone(), xpred.clone())

        baseline_logits = _dlm_decode_logits(
            z, model, 1.0, config, args.self_cond_cfg,
        )
        baseline_ids = baseline_logits.argmax(dim=-1)
        baseline_stats = answer_decoder_stats(
            baseline_logits, answer_starts, answer_lengths, input_ids,
        )

        for state_index in branch_indices:
            z_state, xpred_state = branch_states[state_index]
            z_state_rms = _generated_rms(z_state, cond_mask)
            selfcond_state_rms = _generated_rms(xpred_state, cond_mask)
            for perturb_target in perturb_targets:
                z_branch = z_state.repeat_interleave(replicas, dim=0)
                xpred_branch = xpred_state.repeat_interleave(replicas, dim=0)
                cond_branch = cond_seq.repeat_interleave(replicas, dim=0)
                mask_branch = cond_mask.repeat_interleave(replicas, dim=0)
                ids_branch = input_ids.repeat_interleave(replicas, dim=0)
                starts_branch = answer_starts.repeat_interleave(replicas)
                lengths_branch = answer_lengths.repeat_interleave(replicas)
                z_scale_branch = z_state_rms.repeat_interleave(replicas)
                selfcond_scale_branch = selfcond_state_rms.repeat_interleave(replicas)
                sigma_branch = sigma_pattern.repeat(bsz)
                generated_mask = (1.0 - mask_branch).unsqueeze(-1)
                delta_z = torch.zeros_like(z_branch)
                delta_selfcond = torch.zeros_like(xpred_branch)
                if perturb_target in ("z", "both"):
                    delta_z = torch.randn(
                        z_branch.shape, generator=branch_generator,
                        device=device, dtype=z_branch.dtype,
                    ) * (sigma_branch * z_scale_branch).to(dtype)[:, None, None]
                    delta_z = delta_z * generated_mask
                if perturb_target in ("selfcond", "both"):
                    delta_selfcond = torch.randn(
                        xpred_branch.shape, generator=branch_generator,
                        device=device, dtype=xpred_branch.dtype,
                    ) * (sigma_branch * selfcond_scale_branch).to(dtype)[:, None, None]
                    delta_selfcond = delta_selfcond * generated_mask
                z_branch = restore_cond(
                    z_branch + delta_z, cond_branch, mask_branch,
                )
                xpred_branch = restore_cond(
                    xpred_branch + delta_selfcond, cond_branch, mask_branch,
                )
                z_noise_rms = _generated_rms(delta_z, mask_branch)
                selfcond_noise_rms = _generated_rms(delta_selfcond, mask_branch)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    for step_index in range(state_index, len(t_steps) - 1):
                        z_branch, xpred_branch = _ode_step(
                            model=model, z=z_branch, t=float(t_steps[step_index]),
                            t_next=float(t_steps[step_index + 1]),
                            x_pred_prev=xpred_branch, config=config,
                            cfg_scale=args.cfg,
                            self_cond_cfg_scale=args.self_cond_cfg,
                            cond_seq=cond_branch, cond_seq_mask=mask_branch,
                        )

                logits = _dlm_decode_logits(
                    z_branch, model, 1.0, config, args.self_cond_cfg,
                )
                stats = answer_decoder_stats(
                    logits, starts_branch, lengths_branch, ids_branch,
                )
                prediction_ids = logits.argmax(dim=-1)
                baseline_reference = baseline_ids.repeat_interleave(replicas, dim=0)
                agreement = _answer_agreement(
                    prediction_ids, baseline_reference, starts_branch, lengths_branch,
                )

                for local in range(bsz):
                    meta = batch_meta[local]
                    for sigma_index, sigma in enumerate(sigmas):
                        for branch_id in range(args.branches):
                            expanded = (
                                local * replicas + sigma_index * args.branches + branch_id
                            )
                            per_branch_rows.append({
                                **meta,
                                "perturb_target": perturb_target,
                                "t_index": state_index,
                                "t_state": float(t_steps[state_index]),
                                "sigma": sigma,
                                "branch": branch_id,
                                "z_state_rms": float(z_state_rms[local]),
                                "selfcond_state_rms": float(selfcond_state_rms[local]),
                                "z_noise_rms": float(z_noise_rms[expanded]),
                                "selfcond_noise_rms": float(
                                    selfcond_noise_rms[expanded]
                                ),
                                "baseline_correct": bool(
                                    baseline_stats["correct"][local]
                                ),
                                "correct": bool(stats["correct"][expanded]),
                                "agrees_with_baseline": bool(agreement[expanded]),
                                "prediction_token_id": int(
                                    stats["prediction"][expanded]
                                ),
                            })
        print(f"Branched {cursor}/{len(selected_dataset)} samples", flush=True)

    aggregate = aggregate_rows(per_branch_rows)
    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": checkpoint_step,
        "device": str(device),
        "seed": args.seed,
        "schedule": args.schedule,
        "steps": args.steps,
        "cfg": args.cfg,
        "self_cond_cfg": args.self_cond_cfg,
        "group_specs": specs,
        "selected_source_ids": indices,
        "t_steps": [float(t) for t in t_steps],
        "branch_indices": branch_indices,
        "sigmas": sigmas,
        "perturb_targets": perturb_targets,
        "branches": args.branches,
        "perturbation": "isotropic Gaussian on generated slots; sigma * per-sample z RMS",
        "aggregate": aggregate,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    with (output_dir / "per_branch.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_branch_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    _save_csv(aggregate, output_dir / "stability.csv")
    for perturb_target in perturb_targets:
        _plot(
            aggregate, output_dir, "agreement_rate",
            f"agreement_vs_time_{perturb_target}.png",
            "Agreement with unperturbed answer", perturb_target,
        )
        _plot(
            aggregate, output_dir, "retention_rate_given_baseline_correct",
            f"retention_vs_time_{perturb_target}.png",
            "Correctness retained (baseline-correct only)", perturb_target,
        )
    print(f"Saved branching artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    run(parse_args())
