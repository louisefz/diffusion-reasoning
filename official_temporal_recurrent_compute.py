#!/usr/bin/env python
"""Test when recurrent reasoning compute must be applied along the flow."""

import argparse
import json
import sys
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
    parser.add_argument("--mode", choices=["window", "impulse"], default="window")
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


def make_policies(num_steps, mode):
    policies = {"all_K1": [1] * num_steps, "all_K4": [4] * num_steps}
    if mode == "impulse":
        for step in range(num_steps):
            schedule = [1] * num_steps
            schedule[step] = 4
            policies[f"impulse_{step}"] = schedule
        return policies
    counts = [1, 2, 3, 4, 6, 8, 12]
    for count in counts:
        policies[f"early_{count}"] = [4] * count + [1] * (num_steps - count)
        policies[f"late_{count}"] = [1] * (num_steps - count) + [4] * count
    return policies


def paired_stats(reference, candidate):
    gain = sum((not a) and b for a, b in zip(reference, candidate))
    loss = sum(a and (not b) for a, b in zip(reference, candidate))
    p = binomtest(gain, gain + loss, 0.5, alternative="greater").pvalue if gain + loss else 1.0
    return {"gain": gain, "loss": loss, "net": gain - loss, "p_greater": p}


def plot(rows, output_dir, mode, outer_times):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_name = {row["name"]: row for row in rows}
    if mode == "impulse":
        baseline = 100 * by_name["all_K1"]["accuracy"]
        values = [100 * by_name[f"impulse_{step}"]["accuracy"] for step in range(len(outer_times))]
        gains = [by_name[f"impulse_{step}"]["paired_vs_all_K1"]["gain"] for step in range(len(outer_times))]
        losses = [by_name[f"impulse_{step}"]["paired_vs_all_K1"]["loss"] for step in range(len(outer_times))]
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
        axes[0].plot(outer_times, np.asarray(values) - baseline, "o-", lw=2.3)
        axes[0].axhline(0, color="black", ls=":", lw=1.3)
        axes[0].set(xlabel="Flow time of one K1→K4 compute impulse",
                    ylabel="Endpoint accuracy change (percentage points)",
                    title="Causal compute-response kernel")
        axes[1].plot(outer_times, gains, "o-", lw=2.2, label="wrong → correct")
        axes[1].plot(outer_times, losses, "o-", lw=2.2, label="correct → wrong")
        axes[1].set(xlabel="Flow time of impulse", ylabel="Number of held-out examples",
                    title="Paired endpoint transitions")
        axes[1].legend(frameon=False)
        for ax in axes:
            ax.grid(alpha=0.25)
        fig.savefig(output_dir / "recurrent_compute_response_kernel.png", dpi=220)
        plt.close(fig)
        return
    counts = [1, 2, 3, 4, 6, 8, 12]
    early = [100 * by_name[f"early_{count}"]["accuracy"] for count in counts]
    late = [100 * by_name[f"late_{count}"]["accuracy"] for count in counts]
    full = 100 * by_name["all_K4"]["accuracy"]
    base = 100 * by_name["all_K1"]["accuracy"]
    fig, ax = plt.subplots(figsize=(7.4, 4.5), constrained_layout=True)
    ax.plot(counts, early, "o-", lw=2.5, label="K=4 only in earliest steps")
    ax.plot(counts, late, "o-", lw=2.5, label="K=4 only in latest steps")
    ax.axhline(base, color="gray", ls=":", label=f"K=1 everywhere ({base:.1f}%)")
    ax.axhline(full, color="tab:green", ls="--", label=f"K=4 everywhere ({full:.1f}%)")
    ax.set_xlabel("Number of outer-flow calls receiving K=4 compute")
    ax.set_ylabel("Held-out d4 exact accuracy (%)")
    ax.set_title("When does recurrent reasoning compute matter?")
    ax.set_xticks(counts)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "temporal_recurrent_compute.png", dpi=220)
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
    if model.reasoning_loop_max != 4:
        raise RuntimeError(f"Expected reasoning_loop_max=4, got {model.reasoning_loop_max}")
    dataset, source_ids = select_d4(load_dataset_split(config.eval_data_path), args.samples)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    time_steps = get_sampling_steps(
        args.outer_steps, args.schedule, config.denoiser_p_mean,
        config.denoiser_p_std, device=device, dtype=torch.float32,
    )
    policies = make_policies(args.outer_steps, args.mode)
    results = []
    dtype = next(model.parameters()).dtype

    for name, loop_schedule in policies.items():
        loader = get_dataloader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
            drop_last=False, max_seq_length=config.max_length,
            pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
            max_input_seq_length=config.max_input_length, distributed=False,
        )
        generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
        flags = []
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
            z = (
                torch.randn((take, config.max_length, model.text_encoder_dim),
                            generator=generator, dtype=dtype)
                * config.denoiser_noise_scale
            ).to(device)
            z = restore_cond(z, cond_seq, cond_mask)
            x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_mask)
            use_bf16 = bool(config.use_bf16) and z.is_cuda
            for step_index, (t, t_next) in enumerate(zip(time_steps[:-1], time_steps[1:])):
                t_batch = torch.full((take,), float(t), dtype=z.dtype, device=device)
                loop_counts = torch.full(
                    (take,), loop_schedule[step_index], dtype=torch.long, device=device)
                model_input = torch.cat([z, x_pred], dim=-1)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    net_out, _ = model(
                        model_input, t_batch, deterministic=True,
                        self_cond_cfg_scale=torch.ones_like(t_batch),
                        reasoning_loop_counts=loop_counts,
                    )
                velocity, x_pred = net_out_to_v_x(net_out, z, t_batch, config.t_eps)
                velocity, x_pred = restore_vx(velocity, x_pred, cond_seq, cond_mask)
                z = z + (float(t_next) - float(t)) * velocity
            decoded = _dlm_decode_batch(z, model, 1.0, config, 1.0)
            rows = torch.arange(take, device=device)
            flags.extend((decoded[rows, cond_lengths] == input_ids[rows, cond_lengths]).cpu().tolist())
        result = {
            "name": name,
            "loop_schedule": loop_schedule,
            "accuracy": sum(flags) / len(flags),
            "correct_flags": flags,
        }
        results.append(result)
        print(name, f"accuracy={result['accuracy']:.4f}", flush=True)

    by_name = {row["name"]: row for row in results}
    baseline = by_name["all_K1"]["correct_flags"]
    for row in results:
        row["paired_vs_all_K1"] = paired_stats(baseline, row["correct_flags"])
    report = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": checkpoint_step,
        "samples": args.samples,
        "source_ids": source_ids,
        "outer_times": time_steps[:-1].cpu().tolist(),
        "schedule": args.schedule,
        "mode": args.mode,
        "interpretation": (
            "early_n uses K4 on first n outer calls then K1; late_n is the matched reverse schedule"
            if args.mode == "window" else
            "impulse_i uses K4 only on outer call i and K1 everywhere else"
        ),
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    plot(results, output_dir, args.mode, time_steps[:-1].cpu().tolist())
    print(f"Saved {output_dir}", flush=True)


if __name__ == "__main__":
    main()
