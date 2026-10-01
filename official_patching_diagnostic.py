#!/usr/bin/env python
"""Patch an ELF answer state toward a different-answer trajectory and resume flow."""

import argparse
from collections import defaultdict
import json
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import concatenate_datasets
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
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--branch-states", nargs="+", type=int,
                        default=[0, 1, 4, 8, 12, 16])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.5, 1.0])
    parser.add_argument("--scopes", nargs="+",
                        default=["answer_slot", "target_suffix"])
    parser.add_argument("--modes", nargs="+", default=[
        "z_only", "cache_only", "joint", "z_reset_cache",
    ])
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--fp32", action="store_true",
                        help="Disable BF16 autocast for local-sensitivity measurements")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def select_grouped(dataset, groups, samples_per_group):
    pieces = []
    tasks = dataset["task"]
    depths = dataset["depth"]
    for group in groups:
        indices = [
            index for index, (task, depth) in enumerate(zip(tasks, depths))
            if f"{task}/d{depth}" == group
        ][:samples_per_group]
        if len(indices) != samples_per_group:
            raise ValueError(
                f"Requested {samples_per_group} samples for {group}, found {len(indices)}"
            )
        pieces.append(dataset.select(indices))
    return concatenate_datasets(pieces)


def different_answer_donors(answers, generator):
    size = answers.numel()
    answers_cpu = answers.detach().cpu()
    for _ in range(1000):
        candidates_cpu = torch.randperm(size, generator=generator)
        if torch.all(answers_cpu[candidates_cpu] != answers_cpu):
            return candidates_cpu.to(answers.device)
    base = torch.arange(size, device=answers.device)
    for shift in range(1, size):
        candidates = torch.roll(base, shifts=shift)
        if torch.all(answers[candidates] != answers):
            return candidates
    donors = []
    for index, answer in enumerate(answers.tolist()):
        donor = next(
            (candidate for candidate, value in enumerate(answers.tolist())
             if candidate != index and value != answer),
            None,
        )
        if donor is None:
            raise ValueError("Each batch must contain at least two distinct answers")
        donors.append(donor)
    return torch.tensor(donors, device=answers.device, dtype=torch.long)


def patch_slot(source, donor, rows, slots, alpha):
    result = source.clone()
    source_slot = source[rows, slots]
    donor_slot = donor[rows, slots]
    result[rows, slots] = (1.0 - alpha) * source_slot + alpha * donor_slot
    return result


def patch_suffix(source, donor, source_starts, donor_starts, length, alpha):
    result = source.clone()
    for row in range(source.shape[0]):
        source_start = int(source_starts[row].item())
        donor_start = int(donor_starts[row].item())
        source_slice = source[row, source_start:source_start + length]
        donor_slice = donor[row, donor_start:donor_start + length]
        result[row, source_start:source_start + length] = (
            (1.0 - alpha) * source_slice + alpha * donor_slice
        )
    return result


def suffix_l2(left, right, starts, length):
    distances = []
    for row in range(left.shape[0]):
        start = int(starts[row].item())
        delta = (
            left[row, start:start + length].float()
            - right[row, start:start + length].float()
        )
        distances.append(delta.square().sum().sqrt())
    return torch.stack(distances)


@torch.no_grad()
def run(args, model, encoder, dataset, tokenizer, config, device):
    if args.samples_per_group % args.batch_size:
        raise ValueError("--batch-size must divide --samples-per-group")
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
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    donor_generator = torch.Generator(device="cpu").manual_seed(args.seed + 2)
    baseline = defaultdict(lambda: {"samples": 0, "correct": 0})
    accumulators = defaultdict(lambda: {
        "samples": 0, "source": 0, "donor": 0, "other": 0,
        "correct": 0, "baseline_correct": 0,
        "gain_samples": 0, "input_l2_sum": 0.0,
        "output_l2_sum": 0.0, "gain_sum": 0.0,
        "output_l2_all_samples": 0, "output_l2_all_sum": 0.0,
    })
    seen = 0

    for batch in loader:
        take = len(batch["target"])
        metadata = dataset[seen:seen + take]
        labels = [
            f"{task}/d{depth}"
            for task, depth in zip(metadata["task"], metadata["depth"])
        ]
        if len(set(labels)) != 1:
            raise ValueError(f"A patching batch crossed group boundaries: {set(labels)}")
        group = labels[0]
        input_ids = torch.from_numpy(np.asarray(batch["input_ids"])).to(device).long()
        encoder_mask = torch.from_numpy(
            np.asarray(batch["encoder_attention_mask"])
        ).to(device).float()
        cond_mask = torch.from_numpy(np.asarray(batch["cond_seq_mask"])).to(device).float()
        cond_lengths = cond_mask.to(torch.long).sum(dim=1)
        cond_seq = encode_text(
            input_ids, encoder_mask, encoder, config.latent_mean,
            config.latent_std, use_bf16=bool(config.use_bf16),
        ).to(dtype)
        z = (
            torch.randn(
                (take, config.max_length, model.text_encoder_dim),
                generator=generator, dtype=dtype,
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
        rows = torch.arange(take, device=device)
        final_ids = _dlm_decode_batch(z, model, 1.0, config, 1.0)
        final_answer = final_ids[rows, cond_lengths]
        target_answer = input_ids[rows, cond_lengths]
        baseline_correct = final_answer == target_answer
        baseline[group]["samples"] += take
        baseline[group]["correct"] += baseline_correct.sum().item()
        donor_indices = different_answer_donors(target_answer, donor_generator)
        donor_target = target_answer[donor_indices]
        donor_lengths = cond_lengths[donor_indices]
        generation_length = config.max_length - config.max_input_length

        for state_index in args.branch_states:
            source_z, source_cache = states[state_index]
            donor_z = source_z[donor_indices]
            donor_cache = source_cache[donor_indices]
            for alpha in args.alphas:
                zero_cache = restore_cond(
                    torch.zeros_like(source_cache), cond_seq, cond_mask,
                )
                scopes = {
                    "answer_slot": (
                        patch_slot(source_z, donor_z, rows, cond_lengths, alpha),
                        patch_slot(source_cache, donor_cache, rows, cond_lengths, alpha),
                    ),
                    "target_suffix": (
                        patch_suffix(
                            source_z, donor_z, cond_lengths, donor_lengths,
                            generation_length, alpha,
                        ),
                        patch_suffix(
                            source_cache, donor_cache, cond_lengths, donor_lengths,
                            generation_length, alpha,
                        ),
                    ),
                }
                scopes = {
                    name: value for name, value in scopes.items()
                    if name in args.scopes
                }
                for scope, (patched_z, patched_cache) in scopes.items():
                    modes = {
                        "z_only": (patched_z, source_cache),
                        "cache_only": (source_z, patched_cache),
                        "joint": (patched_z, patched_cache),
                        "z_reset_cache": (patched_z, zero_cache),
                    }
                    modes = {
                        name: value for name, value in modes.items()
                        if name in args.modes
                    }
                    for mode, (branch_z_start, branch_cache_start) in modes.items():
                        input_z_l2 = suffix_l2(
                            branch_z_start, source_z, cond_lengths, generation_length,
                        )
                        input_cache_l2 = suffix_l2(
                            branch_cache_start, source_cache, cond_lengths,
                            generation_length,
                        )
                        input_joint_l2 = (
                            input_z_l2.square() + input_cache_l2.square()
                        ).sqrt()
                        branch_z = branch_z_start.clone()
                        branch_cache = branch_cache_start.clone()
                        with torch.amp.autocast(
                            "cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda" and bool(config.use_bf16),
                        ):
                            for index in range(state_index, args.steps):
                                branch_z, branch_cache = _ode_step(
                                    model=model, z=branch_z, t=t_steps[index].item(),
                                    t_next=t_steps[index + 1].item(),
                                    x_pred_prev=branch_cache, config=config,
                                    cfg_scale=1.0, self_cond_cfg_scale=1.0,
                                    cond_seq=cond_seq, cond_seq_mask=cond_mask,
                                )
                        branch_ids = _dlm_decode_batch(
                            branch_z, model, 1.0, config, 1.0,
                        )
                        answer = branch_ids[rows, cond_lengths]
                        output_l2 = suffix_l2(
                            branch_z, z, cond_lengths, generation_length,
                        )
                        valid_gain = input_joint_l2 > 1e-8
                        gain = output_l2[valid_gain] / input_joint_l2[valid_gain]
                        source_outcome = answer == target_answer
                        donor_outcome = answer == donor_target
                        other_outcome = ~(source_outcome | donor_outcome)
                        key = (group, state_index, float(alpha), scope, mode)
                        values = accumulators[key]
                        values["samples"] += take
                        values["source"] += source_outcome.sum().item()
                        values["donor"] += donor_outcome.sum().item()
                        values["other"] += other_outcome.sum().item()
                        values["correct"] += source_outcome.sum().item()
                        values["baseline_correct"] += baseline_correct.sum().item()
                        values["gain_samples"] += valid_gain.sum().item()
                        values["input_l2_sum"] += input_joint_l2[valid_gain].sum().item()
                        values["output_l2_sum"] += output_l2[valid_gain].sum().item()
                        values["gain_sum"] += gain.sum().item()
                        values["output_l2_all_samples"] += output_l2.numel()
                        values["output_l2_all_sum"] += output_l2.sum().item()
        seen += take

    results = []
    for (group, state_index, alpha, scope, mode), values in sorted(accumulators.items()):
        total = values["samples"]
        results.append({
            "group": group,
            "state_index": state_index,
            "t_state": float(t_steps[state_index]),
            "alpha": alpha,
            "scope": scope,
            "mode": mode,
            "samples": total,
            "source_answer_rate": values["source"] / max(total, 1),
            "donor_answer_rate": values["donor"] / max(total, 1),
            "other_answer_rate": values["other"] / max(total, 1),
            "baseline_accuracy": values["baseline_correct"] / max(total, 1),
            "gain_samples": values["gain_samples"],
            "mean_input_joint_l2": (
                values["input_l2_sum"] / max(values["gain_samples"], 1)
            ),
            "mean_output_l2": (
                values["output_l2_sum"] / max(values["gain_samples"], 1)
            ),
            "mean_directional_gain": (
                values["gain_sum"] / max(values["gain_samples"], 1)
            ),
            "mean_output_l2_all": (
                values["output_l2_all_sum"]
                / max(values["output_l2_all_samples"], 1)
            ),
        })
    return {
        "schedule": "uniform", "steps": args.steps, "cfg": 1.0,
        "samples": seen,
        "baseline": {
            group: {
                "samples": values["samples"],
                "accuracy": values["correct"] / max(values["samples"], 1),
            }
            for group, values in sorted(baseline.items())
        },
        "results": results,
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config_from_yaml(args.config)
    if args.fp32:
        config.use_bf16 = False
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name or config.encoder_model_name
    )
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    dataset = select_grouped(
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
