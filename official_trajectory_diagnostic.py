#!/usr/bin/env python
"""Measure answer accuracy after every ODE update along an ELF trajectory."""

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
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def empty_counts(t_model_values, t_state_values):
    return [dict(
        t_model=float(t_model), t_state=float(t_state),
        xpred_correct=0, z_correct=0, total=0,
        velocity_norm_sum=0.0, velocity_norm_total=0,
        velocity_cosine_sum=0.0, velocity_cosine_total=0,
    ) for t_model, t_state in zip(t_model_values, t_state_values)]


def add_batch_metrics(counts, index, xpred_ids, z_ids, target_ids, cond_lengths,
                      velocity, previous_velocity, row_indices=None):
    if row_indices is None:
        row_indices = torch.arange(target_ids.shape[0], device=target_ids.device)
    if row_indices.numel() == 0:
        return
    rows = torch.arange(target_ids.shape[0], device=target_ids.device)
    cols = cond_lengths.to(torch.long)
    xpred_correct = xpred_ids[rows, cols] == target_ids[rows, cols]
    z_correct = z_ids[rows, cols] == target_ids[rows, cols]
    answer_velocity = velocity[rows, cols].float()
    velocity_norm = answer_velocity.norm(dim=-1)
    counts[index]["xpred_correct"] += xpred_correct[row_indices].sum().item()
    counts[index]["z_correct"] += z_correct[row_indices].sum().item()
    counts[index]["total"] += row_indices.numel()
    counts[index]["velocity_norm_sum"] += velocity_norm[row_indices].sum().item()
    counts[index]["velocity_norm_total"] += row_indices.numel()
    if previous_velocity is not None:
        previous_answer_velocity = previous_velocity[rows, cols].float()
        cosine = torch.nn.functional.cosine_similarity(
            previous_answer_velocity, answer_velocity, dim=-1, eps=1e-8,
        )
        counts[index]["velocity_cosine_sum"] += cosine[row_indices].sum().item()
        counts[index]["velocity_cosine_total"] += row_indices.numel()


def finalize_counts(counts):
    trajectory = []
    for row in counts:
        total = max(row.pop("total"), 1)
        velocity_total = max(row.pop("velocity_norm_total"), 1)
        cosine_total = row.pop("velocity_cosine_total")
        row["xpred_accuracy"] = row.pop("xpred_correct") / total
        row["z_accuracy"] = row.pop("z_correct") / total
        row["velocity_norm"] = row.pop("velocity_norm_sum") / velocity_total
        cosine_sum = row.pop("velocity_cosine_sum")
        row["velocity_cosine_previous"] = (
            cosine_sum / cosine_total if cosine_total else None
        )
        row["samples"] = total
        trajectory.append(row)
    return trajectory


@torch.no_grad()
def run_schedule(model, encoder, dataset, tokenizer, config, device, schedule,
                 steps, cfg, num_samples, batch_size):
    dtype = next(model.parameters()).dtype
    torch.manual_seed(20260909)
    torch.cuda.manual_seed_all(20260909)
    t_steps = get_sampling_steps(
        steps, schedule, config.denoiser_p_mean, config.denoiser_p_std,
        device=device, dtype=dtype,
    )
    counts = empty_counts(t_steps[:-1], t_steps[1:])
    grouped_counts = defaultdict(
        lambda: empty_counts(t_steps[:-1], t_steps[1:])
    )
    generator = torch.Generator(device="cpu").manual_seed(20260909)
    loader = get_dataloader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False,
        max_seq_length=config.max_length,
        pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
        max_input_seq_length=config.max_input_length, distributed=False,
    )
    seen = 0
    for batch in loader:
        if seen >= num_samples:
            break
        take = min(len(batch["target"]), num_samples - seen)
        target_ids = torch.from_numpy(np.asarray(batch["input_ids"][:take])).to(device).long()
        encoder_mask = torch.from_numpy(np.asarray(batch["encoder_attention_mask"][:take])).to(device).float()
        cond_mask = torch.from_numpy(np.asarray(batch["cond_seq_mask"][:take])).to(device).float()
        cond_lengths = cond_mask.to(torch.long).sum(dim=1)
        cond_seq = encode_text(
            target_ids, encoder_mask, encoder, config.latent_mean, config.latent_std,
            use_bf16=bool(config.use_bf16),
        ).to(dtype)
        z = (torch.randn((take, config.max_length, model.text_encoder_dim),
                         generator=generator, dtype=dtype) * config.denoiser_noise_scale).to(device)
        z = restore_cond(z, cond_seq, cond_mask)
        xpred = restore_cond(torch.zeros_like(z), cond_seq, cond_mask)
        previous_velocity = None
        metadata = dataset[seen:seen + take]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda" and bool(config.use_bf16)):
            for index in range(len(t_steps) - 1):
                z_before = z
                dt = float(t_steps[index + 1].item() - t_steps[index].item())
                z, xpred = _ode_step(
                    model=model, z=z, t=t_steps[index].item(),
                    t_next=t_steps[index + 1].item(), x_pred_prev=xpred,
                    config=config, cfg_scale=cfg, self_cond_cfg_scale=1.0,
                    cond_seq=cond_seq, cond_seq_mask=cond_mask,
                )
                velocity = (z - z_before) / dt
                xpred_ids = _dlm_decode_batch(xpred, model, 1.0, config, 1.0)
                z_ids = _dlm_decode_batch(z, model, 1.0, config, 1.0)
                add_batch_metrics(
                    counts, index, xpred_ids, z_ids, target_ids, cond_lengths,
                    velocity, previous_velocity,
                )
                if "task" in metadata and "depth" in metadata:
                    labels = [
                        f"{task}/d{depth}"
                        for task, depth in zip(metadata["task"], metadata["depth"])
                    ]
                    for label in sorted(set(labels)):
                        row_indices = torch.tensor(
                            [i for i, value in enumerate(labels) if value == label],
                            device=device, dtype=torch.long,
                        )
                        add_batch_metrics(
                            grouped_counts[label], index, xpred_ids, z_ids,
                            target_ids, cond_lengths, velocity, previous_velocity,
                            row_indices,
                        )
                previous_velocity = velocity
        seen += take
    return {
        "schedule": schedule,
        "steps": steps,
        "cfg": cfg,
        "samples": seen,
        "trajectory": finalize_counts(counts),
        "groups": {
            label: finalize_counts(grouped_counts[label])
            for label in sorted(grouped_counts)
        },
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config_from_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    dataset = load_dataset_split(config.eval_data_path)
    model, checkpoint_step = load_model(
        config, args.checkpoint, encoder_config, tokenizer, device,
    )
    report = {"checkpoint": args.checkpoint, "checkpoint_step": checkpoint_step,
              "results": []}
    for schedule in ("uniform", "logit_normal"):
        result = run_schedule(
            model, encoder, dataset, tokenizer, config, device, schedule,
            args.steps, args.cfg, args.num_samples, args.batch_size,
        )
        report["results"].append(result)
        print(json.dumps(result), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
