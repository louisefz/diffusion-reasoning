#!/usr/bin/env python
"""Evaluate real variable-K compute allocation at matched inner-step budgets."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from scipy.stats import binomtest
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "official-elf" / "src"))

from configs.config import load_config_from_yaml
from modules.t5_encoder import get_encoder
from official_diagnostics import load_model
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split
from utils.encoder_utils import encode_text
from utils.generation_utils import _dlm_decode_batch
from utils.sampling_utils import get_sampling_steps, net_out_to_v_x, restore_cond, restore_vx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--outer-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--schedule", choices=["logit_normal", "uniform"], default="logit_normal")
    return parser.parse_args()


def select_d4(dataset, count):
    table = dataset.data.table
    mask = pc.and_(
        pc.equal(table.column("task"), pa.scalar("compose")),
        pc.equal(table.column("depth"), pa.scalar(4)),
    )
    indices = pc.indices_nonzero(mask).to_numpy().tolist()[:count]
    if len(indices) < count:
        raise ValueError(f"Requested {count} d4 examples, found {len(indices)}")
    return dataset.select(indices), indices


def policies():
    return {
        "K1_budget16": [1] * 16,
        "uniform_budget32": [2] * 16,
        "early_budget32": [4] * 5 + [2] + [1] * 10,
        "late_budget32": [1] * 10 + [2] + [4] * 5,
        "uniform_budget48": [3] * 16,
        "early_budget48": [4] * 10 + [2] * 2 + [1] * 4,
        "late_budget48": [1] * 4 + [2] * 2 + [4] * 10,
        "K4_budget64": [4] * 16,
    }


def paired(reference, candidate):
    gain = sum((not a) and b for a, b in zip(reference, candidate))
    loss = sum(a and (not b) for a, b in zip(reference, candidate))
    p = binomtest(gain, gain + loss, 0.5, alternative="greater").pvalue if gain + loss else 1.0
    return {"gain": gain, "loss": loss, "net": gain - loss, "p_greater": p}


def plot(rows, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "uniform": "tab:blue", "early": "tab:green", "late": "tab:orange",
        "K1": "gray", "K4": "black",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.3), constrained_layout=True)
    for row in rows:
        prefix = row["name"].split("_", 1)[0]
        axes[0].scatter(row["inner_updates"], 100 * row["accuracy"], s=75,
                        color=colors[prefix], label=prefix)
        axes[0].annotate(row["name"].replace("_budget", " B"),
                         (row["inner_updates"], 100 * row["accuracy"]),
                         xytext=(4, 4), textcoords="offset points", fontsize=7)
        axes[1].scatter(row["sampling_seconds"], 100 * row["accuracy"], s=75,
                        color=colors[prefix])
        axes[1].annotate(row["name"].replace("_budget", " B"),
                         (row["sampling_seconds"], 100 * row["accuracy"]),
                         xytext=(4, 4), textcoords="offset points", fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0].legend(unique.values(), unique.keys(), frameon=False, fontsize=8)
    axes[0].set(xlabel="Inner recurrent updates per trajectory",
                ylabel="Held-out d4 exact accuracy (%)",
                title="Accuracy–compute allocation")
    axes[1].set(xlabel="Measured GPU sampling time (seconds)",
                ylabel="Held-out d4 exact accuracy (%)",
                title="Accuracy–latency trade-off")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(output_dir / "compute_allocation_pareto.png", dpi=220)
    plt.close(fig)


@torch.no_grad()
def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config_from_yaml(args.config)
    config.reasoning_loops = 4
    config.reasoning_loop_max = 4
    config.reasoning_memory_tokens = 4
    config.reasoning_memory_inner_time = True
    config.reasoning_memory_direct_coupling = True
    config.reasoning_memory_bottleneck = True
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    model, checkpoint_step = load_model(config, args.checkpoint, encoder_config, tokenizer, device)
    model.eval()
    dataset, source_ids = select_d4(load_dataset_split(config.eval_data_path), args.samples)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    time_steps = get_sampling_steps(
        args.outer_steps, args.schedule, config.denoiser_p_mean,
        config.denoiser_p_std, device=device, dtype=torch.float32,
    )
    dtype = next(model.parameters()).dtype
    results = []

    # One untimed warm-up removes CUDA initialization from policy timing.
    warm = next(iter(get_dataloader(
        dataset.select(range(min(args.batch_size, len(dataset)))), batch_size=args.batch_size,
        shuffle=False, num_workers=0, drop_last=False, max_seq_length=config.max_length,
        pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
        max_input_seq_length=config.max_input_length, distributed=False,
    )))
    warm_ids = torch.from_numpy(np.asarray(warm["input_ids"])).to(device).long()
    warm_mask = torch.from_numpy(np.asarray(warm["encoder_attention_mask"])).to(device).float()
    warm_cond_mask = torch.from_numpy(np.asarray(warm["cond_seq_mask"])).to(device).float()
    warm_cond = encode_text(warm_ids, warm_mask, encoder, config.latent_mean,
                            config.latent_std, use_bf16=bool(config.use_bf16)).to(dtype)
    warm_z = restore_cond(torch.zeros_like(warm_cond), warm_cond, warm_cond_mask)
    model.reasoning_loops = 4
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        model(torch.cat([warm_z, warm_z], dim=-1), torch.zeros(len(warm_z), device=device, dtype=dtype),
              deterministic=True, self_cond_cfg_scale=torch.ones(len(warm_z), device=device, dtype=dtype))
    if device.type == "cuda":
        torch.cuda.synchronize()

    for name, loop_schedule in policies().items():
        if len(loop_schedule) != args.outer_steps:
            raise ValueError("Policies currently require exactly 16 outer steps")
        loader = get_dataloader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
            drop_last=False, max_seq_length=config.max_length,
            pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
            max_input_seq_length=config.max_input_length, distributed=False,
        )
        generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
        flags = []
        sampling_seconds = 0.0
        for batch in loader:
            take = len(batch["target"])
            input_ids = torch.from_numpy(np.asarray(batch["input_ids"])).to(device).long()
            encoder_mask = torch.from_numpy(np.asarray(batch["encoder_attention_mask"])).to(device).float()
            cond_mask = torch.from_numpy(np.asarray(batch["cond_seq_mask"])).to(device).float()
            cond_lengths = cond_mask.long().sum(dim=1)
            cond_seq = encode_text(
                input_ids, encoder_mask, encoder, config.latent_mean, config.latent_std,
                use_bf16=bool(config.use_bf16),
            ).to(dtype)
            z = (torch.randn((take, config.max_length, model.text_encoder_dim),
                             generator=generator, dtype=dtype) * config.denoiser_noise_scale).to(device)
            z = restore_cond(z, cond_seq, cond_mask)
            x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_mask)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            for step_index, (t, t_next) in enumerate(zip(time_steps[:-1], time_steps[1:])):
                model.reasoning_loops = int(loop_schedule[step_index])
                t_batch = torch.full((take,), float(t), dtype=z.dtype, device=device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=bool(config.use_bf16) and z.is_cuda):
                    net_out, _ = model(
                        torch.cat([z, x_pred], dim=-1), t_batch, deterministic=True,
                        self_cond_cfg_scale=torch.ones_like(t_batch),
                    )
                velocity, x_pred = net_out_to_v_x(net_out, z, t_batch, config.t_eps)
                velocity, x_pred = restore_vx(velocity, x_pred, cond_seq, cond_mask)
                z = z + (float(t_next) - float(t)) * velocity
            if device.type == "cuda":
                torch.cuda.synchronize()
            sampling_seconds += time.perf_counter() - started
            # Hold the native decoder constant at K=4 for every allocation.
            model.reasoning_loops = 4
            decoded = _dlm_decode_batch(z, model, 1.0, config, 1.0)
            rows = torch.arange(take, device=device)
            flags.extend((decoded[rows, cond_lengths] == input_ids[rows, cond_lengths]).cpu().tolist())
        result = {
            "name": name,
            "loop_schedule": loop_schedule,
            "inner_updates": sum(loop_schedule),
            "accuracy": sum(flags) / len(flags),
            "sampling_seconds": sampling_seconds,
            "correct_flags": flags,
        }
        results.append(result)
        print(name, f"updates={sum(loop_schedule)} accuracy={result['accuracy']:.4f} "
              f"sampling_s={sampling_seconds:.4f}", flush=True)

    by_name = {row["name"]: row for row in results}
    for budget in (32, 48):
        uniform = by_name[f"uniform_budget{budget}"]["correct_flags"]
        by_name[f"early_budget{budget}"]["paired_vs_uniform"] = paired(
            uniform, by_name[f"early_budget{budget}"]["correct_flags"])
        by_name[f"late_budget{budget}"]["paired_vs_uniform"] = paired(
            uniform, by_name[f"late_budget{budget}"]["correct_flags"])
        by_name[f"early_budget{budget}"]["paired_vs_late"] = paired(
            by_name[f"late_budget{budget}"]["correct_flags"],
            by_name[f"early_budget{budget}"]["correct_flags"])
    report = {
        "checkpoint": args.checkpoint, "checkpoint_step": checkpoint_step,
        "samples": args.samples, "source_ids": source_ids,
        "schedule": args.schedule, "outer_times": time_steps[:-1].cpu().tolist(),
        "timing_scope": "16 outer flow model calls only; encoding and final K4 decoder excluded",
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    plot(results, output_dir)
    print(f"Saved {output_dir}", flush=True)


if __name__ == "__main__":
    main()
