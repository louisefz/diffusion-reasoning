#!/usr/bin/env python
"""Trace symbolic memory readouts over outer flow time and inner recurrence."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
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
    parser.add_argument("--schedule", default="logit_normal")
    parser.add_argument("--inner-loops", type=int, default=4)
    parser.add_argument("--memory-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260908)
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


def plot(summary, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.asarray(summary["outer_times"])
    matrix = np.asarray(summary["readout_accuracy"])  # time, loop, state
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.8), constrained_layout=True)
    image = None
    for state_index, ax in enumerate(axes):
        image = ax.imshow(
            matrix[:, :, state_index].T,
            origin="lower", aspect="auto", cmap="viridis", vmin=0.125, vmax=1.0,
            extent=[times[0], times[-1], 0.5, matrix.shape[1] + 0.5],
        )
        ax.set_title(f"Readout of true s{state_index + 1}")
        ax.set_xlabel("Outer flow time t")
        ax.set_yticks(range(1, matrix.shape[1] + 1))
        if state_index == 0:
            ax.set_ylabel("Inner recurrent step k")
    fig.colorbar(image, ax=axes, label="Held-out readout accuracy", shrink=0.86)
    fig.suptitle(
        "What recurrent memory represents across flow time and inner reasoning",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(output_dir / "memory_time_by_loop_heatmaps.png", dpi=220)
    plt.close(fig)

    diagonal = np.asarray(summary["aligned_state_accuracy"])
    final_state = matrix[:, :, -1]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)
    for k in range(diagonal.shape[1]):
        axes[0].plot(times, diagonal[:, k], "o-", label=f"k={k + 1} reads s{k + 1}")
        axes[1].plot(times, final_state[:, k], "o-", label=f"memory after k={k + 1}")
    for ax in axes:
        ax.axhline(0.125, color="black", ls=":", lw=1.4, label="chance")
        ax.set_xlabel("Outer flow time t")
        ax.set_ylim(0, 1.03)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Held-out readout accuracy")
    axes[0].set_title("Loop-aligned intermediate states")
    axes[1].set_title("Final answer state s4")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(output_dir / "memory_dynamics_curves.png", dpi=220)
    plt.close(fig)


@torch.no_grad()
def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config_from_yaml(args.config)
    config.reasoning_loops = args.inner_loops
    config.reasoning_memory_tokens = args.memory_tokens
    config.reasoning_memory_inner_time = True
    config.reasoning_memory_direct_coupling = True
    config.reasoning_memory_bottleneck = True

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    model, checkpoint_step = load_model(config, args.checkpoint, encoder_config, tokenizer, device)
    model.eval()

    dataset, source_ids = select_d4(load_dataset_split(config.eval_data_path), args.samples)
    loader = get_dataloader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        drop_last=False, max_seq_length=config.max_length,
        pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
        max_input_seq_length=config.max_input_length, distributed=False,
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    time_steps = get_sampling_steps(
        args.outer_steps, args.schedule, config.denoiser_p_mean,
        config.denoiser_p_std, device=device, dtype=torch.float32,
    )
    call_times = time_steps[:-1]
    correct = torch.zeros(
        len(call_times), args.inner_loops, 4, dtype=torch.float64, device=device)
    true_probability = torch.zeros_like(correct)
    total = 0
    endpoint_correct = 0
    noise_generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    state_token_ids = torch.tensor(config.reasoning_state_token_ids, device=device)
    dtype = next(model.parameters()).dtype

    for batch in loader:
        take = len(batch["target"])
        input_ids = torch.from_numpy(np.asarray(batch["input_ids"])).to(device).long()
        encoder_mask = torch.from_numpy(np.asarray(batch["encoder_attention_mask"])).to(device).float()
        attention_mask = torch.from_numpy(np.asarray(batch["attention_mask"])).to(device).float()
        cond_mask = torch.from_numpy(np.asarray(batch["cond_seq_mask"])).to(device).float()
        cond_lengths = cond_mask.long().sum(dim=1)
        states = torch.as_tensor(np.asarray(batch["states"]), device=device).long()[:, 1:5]
        cond_seq = encode_text(
            input_ids, encoder_mask, encoder, config.latent_mean, config.latent_std,
            use_bf16=bool(config.use_bf16),
        ).to(dtype)
        z = (
            torch.randn((take, config.max_length, model.text_encoder_dim),
                        generator=noise_generator, dtype=dtype)
            * config.denoiser_noise_scale
        ).to(device)
        z = restore_cond(z, cond_seq, cond_mask)
        x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_mask)

        use_bf16 = bool(config.use_bf16) and z.is_cuda
        for time_index, (t, t_next) in enumerate(zip(time_steps[:-1], time_steps[1:])):
            t_batch = torch.full((take,), float(t), dtype=z.dtype, device=device)
            sc_scale = torch.ones((take,), dtype=z.dtype, device=device)
            model_input = torch.cat([z, x_pred], dim=-1)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                net_out, _, reasoning_logits = model(
                    model_input, t_batch, deterministic=True,
                    self_cond_cfg_scale=sc_scale,
                    reasoning_readout_positions=cond_lengths,
                    reasoning_state_token_ids=state_token_ids,
                )
            probabilities = reasoning_logits.float().softmax(dim=-1)
            predictions = probabilities.argmax(dim=-1)
            for state_index in range(4):
                target = states[:, state_index]
                correct[time_index, :, state_index] += (
                    predictions == target[:, None]
                ).sum(dim=0)
                true_probability[time_index, :, state_index] += probabilities.gather(
                    -1,
                    target[:, None, None].expand(-1, args.inner_loops, 1),
                ).squeeze(-1).sum(dim=0)
            velocity, x_pred = net_out_to_v_x(net_out, z, t_batch, config.t_eps)
            velocity, x_pred = restore_vx(velocity, x_pred, cond_seq, cond_mask)
            z = z + (float(t_next) - float(t)) * velocity

        decoded = _dlm_decode_batch(z, model, 1.0, config, 1.0)
        rows = torch.arange(take, device=device)
        endpoint_correct += (decoded[rows, cond_lengths] == input_ids[rows, cond_lengths]).sum().item()
        total += take

    accuracy = (correct / total).cpu().numpy()
    probability = (true_probability / total).cpu().numpy()
    aligned = np.stack([accuracy[:, k, k] for k in range(4)], axis=1)
    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": checkpoint_step,
        "samples": total,
        "source_ids": source_ids,
        "outer_steps": args.outer_steps,
        "schedule": args.schedule,
        "outer_times": call_times.float().cpu().tolist(),
        "inner_loops": args.inner_loops,
        "state_definition": "s_k is the value after applying function k in d4 composition",
        "readout_accuracy": accuracy.tolist(),
        "true_state_probability": probability.tolist(),
        "aligned_state_accuracy": aligned.tolist(),
        "endpoint_accuracy": endpoint_correct / total,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(summary, output_dir)
    print(json.dumps({
        "samples": total,
        "endpoint_accuracy": endpoint_correct / total,
        "final_time_readout_accuracy": accuracy[-1].tolist(),
        "artifacts": str(output_dir),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
