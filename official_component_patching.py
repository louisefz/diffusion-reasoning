#!/usr/bin/env python
"""Patch attention or MLP answer-slot outputs near ELF's causal boundary."""

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
from official_flow_trajectory import answer_decoder_stats
from official_layerwise_patching import donor_indices, semantic_token_map
from official_layerwise_probe import select_balanced
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split
from utils.encoder_utils import encode_text
from utils.generation_utils import _dlm_decode_logits
from utils.sampling_utils import restore_cond


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-group", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--blocks", type=int, nargs="+", default=[9, 10, 11])
    parser.add_argument("--seed", type=int, default=20260919)
    return parser.parse_args()


def component_module(model, block_number, component):
    block = model.blocks[block_number - 1]
    return block.attn if component == "attention" else block.mlp


def capture_components(model, model_input, t_batch, sc_batch, positions,
                       blocks, use_bf16):
    rows = torch.arange(model_input.shape[0], device=model_input.device)
    prefix = (model.num_model_mode_tokens + model.num_time_tokens
              + model.num_self_cond_cfg_tokens)
    captures = {}
    handles = []
    for block in blocks:
        for component in ("attention", "mlp"):
            key = (block, component)

            def hook(_module, _inputs, output, key=key):
                captures[key] = output[rows, positions + prefix].detach().clone()

            handles.append(component_module(model, block, component).register_forward_hook(hook))
    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            endpoint, _ = model(
                model_input, t_batch, deterministic=True,
                self_cond_cfg_scale=sc_batch, decoder_step_active=None,
            )
    finally:
        for handle in handles:
            handle.remove()
    return endpoint, captures


def patched_component_forward(model, model_input, t_batch, sc_batch, positions,
                              block, component, donor_vectors, use_bf16):
    rows = torch.arange(model_input.shape[0], device=model_input.device)
    prefix = (model.num_model_mode_tokens + model.num_time_tokens
              + model.num_self_cond_cfg_tokens)

    def hook(_module, _inputs, output):
        patched = output.clone()
        patched[rows, positions + prefix] = donor_vectors.to(output.dtype)
        return patched

    handle = component_module(model, block, component).register_forward_hook(hook)
    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            endpoint, _ = model(
                model_input, t_batch, deterministic=True,
                self_cond_cfg_scale=sc_batch, decoder_step_active=None,
            )
    finally:
        handle.remove()
    return endpoint


def aggregate(rows):
    buckets = defaultdict(list)
    for row in rows:
        key = (row["group"], row["donor_type"], row["block"], row["component"])
        buckets[key].append(row)
    result = []
    for (group, donor_type, block, component), items in sorted(buckets.items()):
        result.append({
            "group": group,
            "donor_type": donor_type,
            "block": block,
            "component": component,
            "samples": len(items),
            "source_answer_rate": float(np.mean([x["follows_source"] for x in items])),
            "donor_answer_rate": float(np.mean([x["follows_donor"] for x in items])),
            "other_answer_rate": float(np.mean([x["follows_other"] for x in items])),
            "exact_correct_rate": float(np.mean([x["exact_correct"] for x in items])),
        })
    return result


def plot(rows, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = sorted({row["group"] for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for axis, group in zip(axes.flat, groups):
        for component, marker in (("attention", "o"), ("mlp", "s")):
            chosen = [r for r in rows if r["group"] == group
                      and r["donor_type"] == "different"
                      and r["component"] == component]
            axis.plot([r["block"] for r in chosen],
                      [r["donor_answer_rate"] for r in chosen],
                      marker=marker, label=f"{component}: follows donor")
        controls = [r for r in rows if r["group"] == group
                    and r["donor_type"] == "same"]
        axis.plot([r["block"] + (0.05 if r["component"] == "mlp" else -0.05)
                   for r in controls],
                  [r["exact_correct_rate"] for r in controls],
                  ".", color="gray", alpha=0.7, label="same-answer controls")
        axis.set_title(group)
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.25)
    for axis in axes[-1]: axis.set_xlabel("Patched block")
    for axis in axes[:, 0]: axis.set_ylabel("Fraction")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "component_causal_patching.png", dpi=180)
    plt.close(fig)


@torch.no_grad()
def run(args):
    if args.samples_per_group % args.batch_size:
        raise ValueError("--batch-size must divide --samples-per-group")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config_from_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    token_map = semantic_token_map(tokenizer)
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    model, checkpoint_step = load_model(config, args.checkpoint, encoder_config, tokenizer, device)
    if any(block < 1 or block > model.depth for block in args.blocks):
        raise ValueError(f"Blocks must be in [1, {model.depth}]")
    dataset = load_dataset_split(config.eval_data_path)
    selected, expected_groups, source_ids = select_balanced(
        dataset, args.samples_per_group, args.seed,
    )
    loader = get_dataloader(
        selected, batch_size=args.batch_size, shuffle=False, num_workers=0,
        drop_last=False, max_seq_length=config.max_length,
        pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
        max_input_seq_length=config.max_input_length, distributed=False,
    )
    dtype = next(model.parameters()).dtype
    noise_generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    use_bf16 = bool(config.use_bf16) and device.type == "cuda"
    per_sample = []
    cursor = 0
    for batch_index, batch in enumerate(loader):
        bsz = len(batch["target"])
        groups = expected_groups[cursor:cursor + bsz]
        if len(set(groups)) != 1:
            raise ValueError(f"Batch crosses groups: {set(groups)}")
        group = groups[0]
        labels = [int(target) for target in batch["target"]]
        input_ids = torch.from_numpy(np.asarray(batch["input_ids"])).to(device).long()
        encoder_mask = torch.from_numpy(np.asarray(batch["encoder_attention_mask"])).to(device).float()
        cond_mask = torch.from_numpy(np.asarray(batch["cond_seq_mask"])).to(device).float()
        answer_starts = cond_mask.long().sum(dim=1)
        answer_lengths = torch.tensor(
            [len(tokenizer.encode(str(target), add_special_tokens=False)) for target in batch["target"]],
            device=device, dtype=torch.long,
        )
        answer_positions = answer_starts + answer_lengths - 1
        cond_seq = encode_text(input_ids, encoder_mask, encoder, config.latent_mean,
                               config.latent_std, use_bf16=bool(config.use_bf16)).to(dtype)
        z = (torch.randn((bsz, config.max_length, model.text_encoder_dim),
                         generator=noise_generator, dtype=dtype)
             * config.denoiser_noise_scale).to(device)
        z = restore_cond(z, cond_seq, cond_mask)
        previous = restore_cond(torch.zeros_like(z), cond_seq, cond_mask)
        model_input = torch.cat([z, previous], dim=-1)
        t_batch = torch.zeros((bsz,), device=device, dtype=dtype)
        sc_batch = torch.ones((bsz,), device=device, dtype=dtype)
        _baseline, captures = capture_components(
            model, model_input, t_batch, sc_batch, answer_positions,
            args.blocks, use_bf16,
        )
        source_tokens = torch.tensor([token_map[x] for x in labels], device=device)
        for donor_type, same_answer in (("different", False), ("same", True)):
            donors = donor_indices(
                labels, same_answer=same_answer,
                seed=args.seed + batch_index * 17 + int(same_answer),
            ).to(device)
            donor_labels = [labels[int(index)] for index in donors]
            donor_tokens = torch.tensor([token_map[x] for x in donor_labels], device=device)
            for block in args.blocks:
                for component in ("attention", "mlp"):
                    endpoint = patched_component_forward(
                        model, model_input, t_batch, sc_batch, answer_positions,
                        block, component, captures[(block, component)][donors], use_bf16,
                    )
                    logits = _dlm_decode_logits(endpoint, model, 1.0, config, 1.0)
                    stats = answer_decoder_stats(logits, answer_starts, answer_lengths, input_ids)
                    predictions = stats["prediction"]
                    for local in range(bsz):
                        follows_source = bool(predictions[local] == source_tokens[local])
                        follows_donor = bool(predictions[local] == donor_tokens[local])
                        per_sample.append({
                            "source_id": source_ids[cursor + local], "group": group,
                            "source_answer": labels[local], "donor_answer": donor_labels[local],
                            "donor_type": donor_type, "block": block, "component": component,
                            "prediction_token_id": int(predictions[local]),
                            "follows_source": follows_source, "follows_donor": follows_donor,
                            "follows_other": not follows_source and not follows_donor,
                            "exact_correct": bool(stats["correct"][local]),
                        })
        cursor += bsz
        print(f"Patched {cursor}/{len(selected)} samples", flush=True)
    metrics = aggregate(per_sample)
    summary = {
        "checkpoint": args.checkpoint, "checkpoint_step": checkpoint_step,
        "device": str(device), "samples_per_group": args.samples_per_group,
        "batch_size": args.batch_size, "seed": args.seed, "blocks": args.blocks,
        "scope": "attention/MLP output at semantic answer slot, first t=0 denoiser forward",
        "aggregate": metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "per_sample.jsonl").open("w") as handle:
        for row in per_sample: handle.write(json.dumps(row) + "\n")
    with (output_dir / "component_patching_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader(); writer.writerows(metrics)
    plot(metrics, output_dir)
    print(f"Saved component patching artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    run(parse_args())
