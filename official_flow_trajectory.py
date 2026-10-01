#!/usr/bin/env python
"""Record conditional ELF flow trajectories and native-decoder diagnostics.

The recorder stores full latent states and Euler velocities for later causal
branching experiments.  It also produces lightweight per-time diagnostics and
plots so a small run can immediately test whether answer realization shifts
with controlled composition depth.
"""

import argparse
from collections import defaultdict
import csv
import json
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
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split
from utils.encoder_utils import encode_text
from utils.generation_utils import _dlm_decode_logits
from utils.sampling_utils import _ode_step, get_sampling_steps, restore_cond


DEFAULT_GROUPS = ("compose:1:30", "compose:2:30", "compose:4:30", "lookup:4:30")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group", action="append", default=None,
                        help="TASK:DEPTH:COUNT; repeat for balanced groups")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--schedule", choices=("uniform", "logit_normal"),
                        default="logit_normal")
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--self-cond-cfg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--no-save-tensors", action="store_true")
    return parser.parse_args()


def parse_group_spec(spec):
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid group {spec!r}; expected TASK:DEPTH:COUNT")
    task, depth, count = parts
    depth, count = int(depth), int(count)
    if count <= 0:
        raise ValueError(f"Group count must be positive, got {count}")
    return task, depth, count


def select_balanced_indices(dataset, specs, seed):
    """Select deterministic random examples for each requested task/depth."""
    pools = defaultdict(list)
    for index in range(len(dataset)):
        row = dataset[index]
        pools[(str(row["task"]), int(row["depth"]))].append(index)
    rng = random.Random(seed)
    selected, metadata = [], []
    for spec in specs:
        task, depth, count = parse_group_spec(spec)
        candidates = pools[(task, depth)]
        if len(candidates) < count:
            raise ValueError(
                f"Requested {count} examples for {task}/d{depth}, "
                f"but dataset contains {len(candidates)}"
            )
        chosen = sorted(rng.sample(candidates, count))
        for index in chosen:
            row = dataset[index]
            selected.append(index)
            metadata.append({
                "source_id": index,
                "group": f"{task}/d{depth}",
                "task": task,
                "depth": depth,
                "target": str(row["target"]),
                "input": str(row["input"]),
            })
    return selected, metadata


def answer_decoder_stats(logits, answer_starts, answer_lengths, target_ids):
    """Native-decoder exact match plus margin at the answer-bearing token.

    Some one-character targets are multiple SentencePiece tokens (notably
    ``"0"`` under T5: ``["▁", "0"]``). Exact match therefore covers the
    complete target span, while the margin is measured at its final content
    token so groups remain comparable.
    """
    rows = torch.arange(logits.shape[0], device=logits.device)
    answer_positions = answer_starts + answer_lengths - 1
    answer_logits = logits[rows, answer_positions]
    targets = target_ids[rows, answer_positions]
    predictions = answer_logits.argmax(dim=-1)
    all_predictions = logits.argmax(dim=-1)
    exact = torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
    for row in range(logits.shape[0]):
        start = int(answer_starts[row])
        stop = start + int(answer_lengths[row])
        exact[row] = all_predictions[row, start:stop].eq(
            target_ids[row, start:stop]
        ).all()
    target_logits = answer_logits.gather(1, targets[:, None]).squeeze(1)
    masked = answer_logits.clone()
    masked[rows, targets] = -torch.inf
    best_other = masked.max(dim=-1).values
    top2 = answer_logits.topk(k=2, dim=-1).values
    return {
        "prediction": predictions,
        "target": targets,
        "correct": exact,
        "target_margin": target_logits - best_other,
        "top1_margin": top2[:, 0] - top2[:, 1],
    }


def _empty_group_rows(groups, t_steps):
    return {
        group: [dict(
            t_state=float(t), samples=0, z_correct=0, xpred_correct=0,
            xpred_total=0, z_target_margin_sum=0.0,
            xpred_target_margin_sum=0.0, velocity_norm_sum=0.0,
            velocity_total=0, velocity_cosine_sum=0.0,
            velocity_cosine_total=0,
        ) for t in t_steps]
        for group in groups
    }


def _update_aggregate(row, z_stats, xpred_stats, velocity, previous_velocity,
                      answer_positions, indices):
    if not indices:
        return
    idx = torch.tensor(indices, device=z_stats["correct"].device, dtype=torch.long)
    row["samples"] += len(indices)
    row["z_correct"] += z_stats["correct"][idx].sum().item()
    row["z_target_margin_sum"] += z_stats["target_margin"][idx].float().sum().item()
    if xpred_stats is not None:
        row["xpred_correct"] += xpred_stats["correct"][idx].sum().item()
        row["xpred_total"] += len(indices)
        row["xpred_target_margin_sum"] += xpred_stats["target_margin"][idx].float().sum().item()
    if velocity is not None:
        rows = torch.arange(velocity.shape[0], device=velocity.device)
        answer_v = velocity[rows, answer_positions].float()
        norms = answer_v.norm(dim=-1)
        row["velocity_norm_sum"] += norms[idx].sum().item()
        row["velocity_total"] += len(indices)
        if previous_velocity is not None:
            previous_v = previous_velocity[rows, answer_positions].float()
            cosine = torch.nn.functional.cosine_similarity(
                previous_v, answer_v, dim=-1, eps=1e-8,
            )
            row["velocity_cosine_sum"] += cosine[idx].sum().item()
            row["velocity_cosine_total"] += len(indices)


def _finalize(rows_by_group):
    result = {}
    for group, rows in rows_by_group.items():
        result[group] = []
        for raw in rows:
            row = dict(raw)
            samples = row.pop("samples")
            xpred_total = row.pop("xpred_total")
            velocity_total = row.pop("velocity_total")
            cosine_total = row.pop("velocity_cosine_total")
            xpred_correct = row.pop("xpred_correct")
            xpred_margin_sum = row.pop("xpred_target_margin_sum")
            velocity_norm_sum = row.pop("velocity_norm_sum")
            velocity_cosine_sum = row.pop("velocity_cosine_sum")
            row["samples"] = samples
            row["z_accuracy"] = row.pop("z_correct") / max(samples, 1)
            row["xpred_accuracy"] = (
                xpred_correct / xpred_total if xpred_total else None
            )
            row["z_target_margin"] = row.pop("z_target_margin_sum") / max(samples, 1)
            row["xpred_target_margin"] = (
                xpred_margin_sum / xpred_total
                if xpred_total else None
            )
            row["velocity_norm"] = (
                velocity_norm_sum / velocity_total
                if velocity_total else None
            )
            row["velocity_cosine_previous"] = (
                velocity_cosine_sum / cosine_total
                if cosine_total else None
            )
            result[group].append(row)
    return result


def _save_csv(summary, path):
    fieldnames = [
        "group", "t_state", "samples", "z_accuracy", "xpred_accuracy",
        "z_target_margin", "xpred_target_margin", "velocity_norm",
        "velocity_cosine_previous",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group, rows in summary["groups"].items():
            for row in rows:
                writer.writerow({"group": group, **row})


def _plot(summary, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "compose/d1": "#1f77b4", "compose/d2": "#ff7f0e",
        "compose/d4": "#d62728", "lookup/d4": "#2ca02c",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for group, rows in summary["groups"].items():
        times = [row["t_state"] for row in rows]
        color = colors.get(group)
        axes[0].plot(times, [row["z_accuracy"] for row in rows], marker="o",
                     markersize=3, label=group, color=color)
        xpred_rows = [row for row in rows if row["xpred_accuracy"] is not None]
        axes[1].plot(
            [row["t_state"] for row in xpred_rows],
            [row["xpred_accuracy"] for row in xpred_rows],
            marker="o", markersize=3, label=group, color=color,
        )
    axes[0].set_title("Current latent $z_t$")
    axes[1].set_title("Predicted endpoint $\\hat{x}_t$")
    for axis in axes:
        axis.set_xlabel("Flow time")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Native-decoder answer accuracy")
    axes[1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_flow.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for group, rows in summary["groups"].items():
        color = colors.get(group)
        valid = [row for row in rows if row["velocity_norm"] is not None]
        axes[0].plot([row["t_state"] for row in valid],
                     [row["velocity_norm"] for row in valid], marker="o",
                     markersize=3, label=group, color=color)
        valid_cos = [row for row in rows if row["velocity_cosine_previous"] is not None]
        axes[1].plot([row["t_state"] for row in valid_cos],
                     [row["velocity_cosine_previous"] for row in valid_cos], marker="o",
                     markersize=3, label=group, color=color)
    axes[0].set_title("Answer-slot velocity norm")
    axes[1].set_title("Velocity direction consistency")
    for axis in axes:
        axis.set_xlabel("Flow time")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Norm")
    axes[1].set_ylabel("Cosine with previous step")
    axes[1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "velocity_vs_flow.png", dpi=180)
    plt.close(fig)


@torch.no_grad()
def record(args):
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
    noise_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    loader = get_dataloader(
        selected_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, drop_last=False, max_seq_length=config.max_length,
        pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
        max_input_seq_length=config.max_input_length, distributed=False,
    )

    group_names = [f"{task}/d{depth}" for task, depth, _ in map(parse_group_spec, specs)]
    aggregates = _empty_group_rows(group_names, [float(t) for t in t_steps])
    tensor_batches = []
    sample_rows = []
    cursor = 0

    for batch in loader:
        bsz = len(batch["target"])
        batch_meta = metadata[cursor:cursor + bsz]
        cursor += bsz
        input_ids = torch.from_numpy(np.asarray(batch["input_ids"])).to(device).long()
        encoder_mask = torch.from_numpy(np.asarray(batch["encoder_attention_mask"])).to(device).float()
        cond_mask = torch.from_numpy(np.asarray(batch["cond_seq_mask"])).to(device).float()
        answer_starts = cond_mask.to(torch.long).sum(dim=1)
        answer_lengths = torch.tensor(
            [len(tokenizer.encode(str(target), add_special_tokens=False))
             for target in batch["target"]],
            device=device, dtype=torch.long,
        )
        answer_positions = answer_starts + answer_lengths - 1
        cond_seq = encode_text(
            input_ids, encoder_mask, encoder, config.latent_mean,
            config.latent_std, use_bf16=bool(config.use_bf16),
        ).to(dtype)
        z = (torch.randn(
            (bsz, config.max_length, model.text_encoder_dim),
            generator=noise_generator, dtype=dtype,
        ) * config.denoiser_noise_scale).to(device)
        z = restore_cond(z, cond_seq, cond_mask)
        xpred = restore_cond(torch.zeros_like(z), cond_seq, cond_mask)

        z_history = [z.detach().to("cpu", dtype=torch.bfloat16)]
        velocity_history = []
        previous_velocity = None

        def evaluate_state(time_index, current_z, current_xpred, velocity=None):
            z_logits = _dlm_decode_logits(
                current_z, model, 1.0, config, args.self_cond_cfg,
            )
            z_stats = answer_decoder_stats(
                z_logits, answer_starts, answer_lengths, input_ids,
            )
            xpred_stats = None
            if current_xpred is not None:
                xpred_logits = _dlm_decode_logits(
                    current_xpred, model, 1.0, config, args.self_cond_cfg,
                )
                xpred_stats = answer_decoder_stats(
                    xpred_logits, answer_starts, answer_lengths, input_ids,
                )
            local_groups = defaultdict(list)
            for local_index, meta in enumerate(batch_meta):
                local_groups[meta["group"]].append(local_index)
                row = {
                    **meta,
                    "t_index": time_index,
                    "t_state": float(t_steps[time_index]),
                    "target_token_id": int(z_stats["target"][local_index]),
                    "z_prediction_token_id": int(z_stats["prediction"][local_index]),
                    "z_correct": bool(z_stats["correct"][local_index]),
                    "z_target_margin": float(z_stats["target_margin"][local_index]),
                    "z_top1_margin": float(z_stats["top1_margin"][local_index]),
                }
                if xpred_stats is not None:
                    row.update({
                        "xpred_prediction_token_id": int(xpred_stats["prediction"][local_index]),
                        "xpred_correct": bool(xpred_stats["correct"][local_index]),
                        "xpred_target_margin": float(xpred_stats["target_margin"][local_index]),
                        "xpred_top1_margin": float(xpred_stats["top1_margin"][local_index]),
                    })
                sample_rows.append(row)
            for group, local_indices in local_groups.items():
                _update_aggregate(
                    aggregates[group][time_index], z_stats, xpred_stats,
                    velocity, previous_velocity, answer_positions, local_indices,
                )

        # Initial Gaussian state has no model-predicted endpoint yet.
        evaluate_state(0, z, None)
        use_bf16 = bool(config.use_bf16) and device.type == "cuda"
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            for step_index in range(len(t_steps) - 1):
                z_before = z
                t = float(t_steps[step_index])
                t_next = float(t_steps[step_index + 1])
                z, xpred = _ode_step(
                    model=model, z=z, t=t, t_next=t_next,
                    x_pred_prev=xpred, config=config, cfg_scale=args.cfg,
                    self_cond_cfg_scale=args.self_cond_cfg,
                    cond_seq=cond_seq, cond_seq_mask=cond_mask,
                )
                velocity = (z - z_before) / (t_next - t)
                velocity_history.append(
                    velocity.detach().to("cpu", dtype=torch.bfloat16)
                )
                z_history.append(z.detach().to("cpu", dtype=torch.bfloat16))
                evaluate_state(step_index + 1, z, xpred, velocity)
                previous_velocity = velocity

        if not args.no_save_tensors:
            tensor_batches.append({
                "source_ids": torch.tensor([m["source_id"] for m in batch_meta]),
                "answer_starts": answer_starts.detach().cpu(),
                "answer_lengths": answer_lengths.detach().cpu(),
                "answer_positions": answer_positions.detach().cpu(),
                "target_token_ids": input_ids[
                    torch.arange(bsz, device=device), answer_positions
                ].detach().cpu(),
                "z": torch.stack(z_history, dim=1),
                "velocity": torch.stack(velocity_history, dim=1),
            })
        print(f"Recorded {cursor}/{len(selected_dataset)} samples", flush=True)

    groups = _finalize(aggregates)
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
        "groups": groups,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    with (output_dir / "per_sample.jsonl").open("w", encoding="utf-8") as handle:
        for row in sample_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    _save_csv(summary, output_dir / "trajectory.csv")
    _plot(summary, output_dir)

    if not args.no_save_tensors:
        tensor_payload = {
            "checkpoint": args.checkpoint,
            "checkpoint_step": checkpoint_step,
            "t_steps": t_steps.detach().float().cpu(),
            "groups": [m["group"] for m in metadata],
            "source_ids": torch.cat([batch["source_ids"] for batch in tensor_batches]),
            "answer_starts": torch.cat([batch["answer_starts"] for batch in tensor_batches]),
            "answer_lengths": torch.cat([batch["answer_lengths"] for batch in tensor_batches]),
            "answer_positions": torch.cat([batch["answer_positions"] for batch in tensor_batches]),
            "target_token_ids": torch.cat([batch["target_token_ids"] for batch in tensor_batches]),
            "z": torch.cat([batch["z"] for batch in tensor_batches], dim=0),
            "velocity": torch.cat([batch["velocity"] for batch in tensor_batches], dim=0),
        }
        torch.save(tensor_payload, output_dir / "trajectory_tensors.pt")
        # Make the batch job self-verifying: malformed or incomplete artifacts
        # fail the Slurm job instead of being mistaken for a successful run.
        from verify_flow_trajectory import validate
        validate(output_dir)
    print(json.dumps({group: rows[-1] for group, rows in groups.items()}, indent=2), flush=True)
    print(f"Saved trajectory artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    record(parse_args())
