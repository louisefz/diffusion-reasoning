#!/usr/bin/env python
"""Measure local answer stability by perturbing and resuming ELF trajectories."""

import argparse
from collections import defaultdict
import json
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
from utils.generation_utils import _dlm_decode_batch
from utils.sampling_utils import _ode_step, get_sampling_steps, restore_cond


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--groups", nargs="+", default=[
        "lookup/d1", "compose/d1", "compose/d2",
    ])
    parser.add_argument("--samples-per-group", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--branch-states", nargs="+", type=int,
                        default=[1, 4, 8, 12, 16])
    parser.add_argument("--sigmas", nargs="+", type=float,
                        default=[0.0, 0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def select_balanced(dataset, requested_groups, samples_per_group):
    selected = []
    counts = defaultdict(int)
    requested = set(requested_groups)
    for index, (task, depth) in enumerate(zip(dataset["task"], dataset["depth"])):
        group = f"{task}/d{depth}"
        if group in requested and counts[group] < samples_per_group:
            selected.append(index)
            counts[group] += 1
        if all(counts[group] >= samples_per_group for group in requested):
            break
    missing = {
        group: samples_per_group - counts[group]
        for group in requested if counts[group] < samples_per_group
    }
    if missing:
        raise ValueError(f"Not enough samples for requested groups: {missing}")
    return dataset.select(selected)


def update(accumulators, key, correct, agreement, baseline_correct, delta_norm):
    row = accumulators[key]
    row["samples"] += correct.numel()
    row["correct"] += correct.sum().item()
    row["agreement"] += agreement.sum().item()
    row["baseline_correct"] += baseline_correct.sum().item()
    row["correct_given_baseline_correct"] += (
        correct & baseline_correct
    ).sum().item()
    row["delta_norm_sum"] += delta_norm.sum().item()


@torch.no_grad()
def run(args, model, encoder, dataset, tokenizer, config, device):
    if any(index < 0 or index > args.steps for index in args.branch_states):
        raise ValueError("branch state indices must be between 0 and --steps")
    dtype = next(model.parameters()).dtype
    t_steps = get_sampling_steps(
        args.steps, "uniform", config.denoiser_p_mean, config.denoiser_p_std,
        device=device, dtype=dtype,
    )
    loader = get_dataloader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        drop_last=False, max_seq_length=config.max_length,
        pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
        max_input_seq_length=config.max_input_length, distributed=False,
    )
    initial_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    perturb_generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    accumulators = defaultdict(lambda: {
        "samples": 0, "correct": 0, "agreement": 0,
        "baseline_correct": 0, "correct_given_baseline_correct": 0,
        "delta_norm_sum": 0.0,
    })
    baseline_accumulators = defaultdict(lambda: {"samples": 0, "correct": 0})
    seen = 0

    for batch in loader:
        take = len(batch["target"])
        target_ids = torch.from_numpy(np.asarray(batch["input_ids"])).to(device).long()
        encoder_mask = torch.from_numpy(
            np.asarray(batch["encoder_attention_mask"])
        ).to(device).float()
        cond_mask = torch.from_numpy(np.asarray(batch["cond_seq_mask"])).to(device).float()
        cond_lengths = cond_mask.to(torch.long).sum(dim=1)
        cond_seq = encode_text(
            target_ids, encoder_mask, encoder, config.latent_mean,
            config.latent_std, use_bf16=bool(config.use_bf16),
        ).to(dtype)
        z = (
            torch.randn(
                (take, config.max_length, model.text_encoder_dim),
                generator=initial_generator, dtype=dtype,
            ) * config.denoiser_noise_scale
        ).to(device)
        z = restore_cond(z, cond_seq, cond_mask)
        xpred = restore_cond(torch.zeros_like(z), cond_seq, cond_mask)
        states = [(z.clone(), xpred.clone())]
        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16,
            enabled=device.type == "cuda" and bool(config.use_bf16),
        ):
            for index in range(args.steps):
                z, xpred = _ode_step(
                    model=model, z=z, t=t_steps[index].item(),
                    t_next=t_steps[index + 1].item(), x_pred_prev=xpred,
                    config=config, cfg_scale=1.0, self_cond_cfg_scale=1.0,
                    cond_seq=cond_seq, cond_seq_mask=cond_mask,
                )
                states.append((z.clone(), xpred.clone()))
        baseline_ids = _dlm_decode_batch(z, model, 1.0, config, 1.0)
        rows = torch.arange(take, device=device)
        baseline_answer = baseline_ids[rows, cond_lengths]
        target_answer = target_ids[rows, cond_lengths]
        baseline_correct = baseline_answer == target_answer
        metadata = dataset[seen:seen + take]
        labels = [
            f"{task}/d{depth}"
            for task, depth in zip(metadata["task"], metadata["depth"])
        ]
        for row_index, label in enumerate(labels):
            baseline_accumulators[label]["samples"] += 1
            baseline_accumulators[label]["correct"] += int(baseline_correct[row_index])

        for state_index in args.branch_states:
            state_z, state_xpred = states[state_index]
            for sigma in args.sigmas:
                expanded_z = state_z.repeat_interleave(args.replicas, dim=0)
                expanded_cond = cond_seq.repeat_interleave(args.replicas, dim=0)
                expanded_mask = cond_mask.repeat_interleave(args.replicas, dim=0)
                expanded_lengths = cond_lengths.repeat_interleave(args.replicas)
                expanded_target = target_answer.repeat_interleave(args.replicas)
                expanded_baseline = baseline_answer.repeat_interleave(args.replicas)
                expanded_baseline_correct = baseline_correct.repeat_interleave(args.replicas)
                expanded_xpred = state_xpred.repeat_interleave(args.replicas, dim=0)
                delta = torch.zeros_like(expanded_z)
                noise = torch.randn(
                    (expanded_z.shape[0], expanded_z.shape[-1]),
                    generator=perturb_generator, dtype=expanded_z.dtype,
                ).to(device) * sigma
                expanded_rows = torch.arange(expanded_z.shape[0], device=device)
                delta[expanded_rows, expanded_lengths] = noise
                perturbed_z = expanded_z + delta
                delta_norm = noise.float().norm(dim=-1)

                for mode in ("retain_cache", "reset_cache"):
                    branch_z = perturbed_z.clone()
                    if mode == "retain_cache":
                        branch_xpred = expanded_xpred.clone()
                    else:
                        branch_xpred = restore_cond(
                            torch.zeros_like(expanded_xpred), expanded_cond, expanded_mask,
                        )
                    with torch.amp.autocast(
                        "cuda", dtype=torch.bfloat16,
                        enabled=device.type == "cuda" and bool(config.use_bf16),
                    ):
                        for index in range(state_index, args.steps):
                            branch_z, branch_xpred = _ode_step(
                                model=model, z=branch_z, t=t_steps[index].item(),
                                t_next=t_steps[index + 1].item(),
                                x_pred_prev=branch_xpred, config=config,
                                cfg_scale=1.0, self_cond_cfg_scale=1.0,
                                cond_seq=expanded_cond, cond_seq_mask=expanded_mask,
                            )
                    branch_ids = _dlm_decode_batch(
                        branch_z, model, 1.0, config, 1.0,
                    )
                    branch_answer = branch_ids[expanded_rows, expanded_lengths]
                    correct = branch_answer == expanded_target
                    agreement = branch_answer == expanded_baseline
                    expanded_labels = [
                        label for label in labels for _ in range(args.replicas)
                    ]
                    for label in sorted(set(expanded_labels)):
                        group_indices = torch.tensor(
                            [i for i, value in enumerate(expanded_labels) if value == label],
                            device=device, dtype=torch.long,
                        )
                        key = (label, state_index, float(sigma), mode)
                        update(
                            accumulators, key, correct[group_indices],
                            agreement[group_indices],
                            expanded_baseline_correct[group_indices],
                            delta_norm[group_indices],
                        )
        seen += take

    results = []
    for (group, state_index, sigma, mode), values in sorted(accumulators.items()):
        samples = values["samples"]
        baseline_correct_count = values["baseline_correct"]
        results.append({
            "group": group,
            "state_index": state_index,
            "t_state": float(t_steps[state_index]),
            "sigma_per_coordinate": sigma,
            "mode": mode,
            "samples": samples,
            "accuracy": values["correct"] / max(samples, 1),
            "agreement_with_baseline": values["agreement"] / max(samples, 1),
            "accuracy_given_baseline_correct": (
                values["correct_given_baseline_correct"]
                / max(baseline_correct_count, 1)
            ),
            "baseline_correct_samples": baseline_correct_count,
            "mean_perturbation_l2": values["delta_norm_sum"] / max(samples, 1),
        })
    return {
        "schedule": "uniform", "steps": args.steps, "cfg": 1.0,
        "samples": seen, "replicas": args.replicas,
        "baseline": {
            group: {
                "samples": values["samples"],
                "accuracy": values["correct"] / max(values["samples"], 1),
            }
            for group, values in sorted(baseline_accumulators.items())
        },
        "results": results,
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config_from_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name or config.encoder_model_name
    )
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    dataset = select_balanced(
        load_dataset_split(config.eval_data_path), args.groups,
        args.samples_per_group,
    )
    model, checkpoint_step = load_model(
        config, args.checkpoint, encoder_config, tokenizer, device,
    )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    report = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": checkpoint_step,
        "groups": args.groups,
        "samples_per_group": args.samples_per_group,
        "diagnostic": run(
            args, model, encoder, dataset, tokenizer, config, device,
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["diagnostic"]["baseline"], indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
