#!/usr/bin/env python
"""Causally trace reasoning information into ELF velocity and answer-state motion.

For minimal source/donor composition pairs, patch block-11 head-3 K or V at one
flow time, measure the induced answer-position velocity shift, its movement
toward the paired donor trajectory/endpoint, and the final counterfactual answer.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "official-elf" / "src"))

from configs.config import load_config_from_yaml
from modules.t5_encoder import get_encoder
from official_counterfactual_patching import build_pairs
from official_diagnostics import load_model
from official_flow_layer_causal_map import decode_state, prepare_flow
from official_layerwise_patching import semantic_token_map
from official_temporal_causal_tracing import capture_qkv
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split
from utils.sampling_utils import _forward_sample, get_sampling_steps


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True); p.add_argument("--pairs-per-step", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16); p.add_argument("--flow-steps", type=int, default=8)
    p.add_argument("--cfg", type=float, default=2.); p.add_argument("--self-cond-cfg", type=float, default=1.)
    p.add_argument("--block", type=int, default=11); p.add_argument("--head", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260926)
    return p.parse_args()


def forward_velocity(model, state, t, config, cfg, self_cond_cfg):
    t_batch = torch.full((state["z"].shape[0],), float(t), dtype=state["z"].dtype,
                         device=state["z"].device)
    return _forward_sample(
        model=model, z=state["z"], t_batch=t_batch, x_pred_prev=state["previous"],
        config=config, cfg_scale=cfg, self_cond_cfg_scale=self_cond_cfg,
        cond_seq=state["cond_seq"], cond_seq_mask=state["cond_mask"],
    )


def patched_velocity(model, state, replacement_qkv, component, t, config,
                     cfg, self_cond_cfg, block, head):
    attention = model.blocks[block - 1].attn
    module = attention.qkv
    component_index = {"k": 1, "v": 2}[component]
    calls = 0

    def hook(_module, _inputs, output):
        nonlocal calls
        current = calls; calls += 1
        if current != 0:  # conditional CFG call is first
            return output
        bsz, length, _ = output.shape
        patched = output.clone().reshape(
            bsz, length, 3, attention.num_heads, attention.dim // attention.num_heads)
        replacement = replacement_qkv.reshape_as(patched)
        patched[:, :, component_index, head] = replacement[:, :, component_index, head].to(patched.dtype)
        return patched.reshape_as(output)

    handle = module.register_forward_hook(hook)
    try:
        velocity, prediction = forward_velocity(model, state, t, config, cfg, self_cond_cfg)
    finally:
        handle.remove()
    expected = 1 if cfg == 1. else 2
    if calls != expected: raise RuntimeError(f"Expected {expected} QKV calls, got {calls}")
    return velocity, prediction


def advance(state, velocity, prediction, dt):
    result = dict(state)
    result["z"] = state["z"] + dt * velocity
    result["previous"] = prediction
    return result


def finish(model, state, start_index, t_steps, config, cfg, self_cond_cfg):
    for index in range(start_index, len(t_steps) - 1):
        velocity, prediction = forward_velocity(model, state, t_steps[index], config, cfg, self_cond_cfg)
        state = advance(state, velocity, prediction, t_steps[index + 1] - t_steps[index])
    return state


def cosine(a, b):
    return F.cosine_similarity(a.float(), b.float(), dim=-1, eps=1e-8)


def aggregate(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["component"], row["intervention_step"], row["time_index"])].append(row)
    metrics = []
    for (component, edit_step, time_index), items in sorted(buckets.items()):
        eligible = [x for x in items if x["eligible"]]
        relevant = [x for x in eligible if x["mechanistically_relevant"]]
        chosen = relevant or eligible or items
        metrics.append({
            "component": component, "intervention_step": edit_step,
            "time_index": time_index, "flow_time": items[0]["flow_time"],
            "samples": len(items), "eligible_samples": len(eligible),
            "relevant_samples": len(relevant),
            "velocity_shift_norm": float(np.mean([x["velocity_shift_norm"] for x in chosen])),
            "velocity_endpoint_alignment": float(np.mean([x["velocity_endpoint_alignment"] for x in chosen])),
            "velocity_local_donor_alignment": float(np.mean([x["velocity_local_donor_alignment"] for x in chosen])),
            "one_step_donor_progress": float(np.mean([x["one_step_donor_progress"] for x in chosen])),
            "endpoint_projection": float(np.mean([x["endpoint_projection"] for x in chosen])),
            "counterfactual_answer_rate": float(np.mean([x["follows_counterfactual"] for x in chosen])),
            "original_answer_rate": float(np.mean([x["follows_original"] for x in chosen])),
        })
    return metrics


def plot(rows, output_dir, flow_steps, t_steps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    relevant = {
        "k": [r for r in rows if r["component"] == "k" and r["intervention_step"] in (1, 2, 3)
              and r["eligible"]],
        "v": [r for r in rows if r["component"] == "v" and r["intervention_step"] == 4
              and r["eligible"]],
    }
    fields = [
        ("velocity_endpoint_alignment", "Velocity shift alignment\nwith donor endpoint"),
        ("one_step_donor_progress", "Normalized one-step progress\ntoward donor trajectory"),
        ("endpoint_projection", "Flow-step displacement along\ndonor endpoint direction"),
        ("follows_counterfactual", "Final donor-answer rate"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    x = np.arange(flow_steps)
    labels = [f"{float(t_steps[i]):.3g}" for i in range(flow_steps)]
    for ri, component in enumerate(("k", "v")):
        chosen = relevant[component]
        for ci, (field, title) in enumerate(fields):
            values = [np.mean([r[field] for r in chosen if r["time_index"] == i])
                      for i in range(flow_steps)]
            axes[ri, ci].plot(x, values, marker="o", linewidth=2)
            axes[ri, ci].axhline(0, color="black", linewidth=.8, alpha=.5)
            axes[ri, ci].grid(alpha=.25); axes[ri, ci].set_title(title)
            axes[ri, ci].set_xticks(x, labels, rotation=30)
            if ci == 0: axes[ri, ci].set_ylabel(f"Head-3 {component.upper()}")
    fig.supxlabel("Flow intervention time")
    fig.tight_layout(); fig.savefig(output_dir / "reasoning_velocity_transport.png", dpi=220)
    plt.close(fig)


@torch.no_grad()
def run(args):
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config_from_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    token_map = semantic_token_map(tokenizer)
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    model, checkpoint_step = load_model(config, args.checkpoint, encoder_config, tokenizer, device)
    validation = load_dataset_split(config.eval_data_path)
    originals, counterfactuals, counts = build_pairs(validation, tokenizer, args.pairs_per_step, args.seed)
    loader_args = dict(batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False,
        max_seq_length=config.max_length, pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
        max_input_seq_length=config.max_input_length, distributed=False)
    source_loader = get_dataloader(originals, **loader_args)
    donor_loader = get_dataloader(counterfactuals, **loader_args)
    dtype = next(model.parameters()).dtype
    t_steps = get_sampling_steps(args.flow_steps, "uniform", config.denoiser_p_mean,
        config.denoiser_p_std, device=device, dtype=dtype)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    use_bf16 = bool(config.use_bf16) and device.type == "cuda"
    rows = []; cursor = 0

    for source_batch, donor_batch in zip(source_loader, donor_loader):
        bsz = len(source_batch["target"])
        noise = torch.randn((bsz, config.max_length, model.text_encoder_dim),
                            generator=generator, dtype=dtype)
        source = prepare_flow(source_batch, noise, model, encoder, tokenizer, config, device)
        donor = prepare_flow(donor_batch, noise, model, encoder, tokenizer, config, device)
        source_states = []; donor_states = []; source_velocities = []; donor_velocities = []; donor_qkvs = []
        for index in range(args.flow_steps):
            source_states.append(source); donor_states.append(donor)
            donor_qkvs.append(capture_qkv(model, donor, t_steps[index], args.block,
                                           args.self_cond_cfg, use_bf16))
            source_v, source_x = forward_velocity(model, source, t_steps[index], config,
                                                   args.cfg, args.self_cond_cfg)
            donor_v, donor_x = forward_velocity(model, donor, t_steps[index], config,
                                                 args.cfg, args.self_cond_cfg)
            source_velocities.append(source_v); donor_velocities.append(donor_v)
            dt = t_steps[index + 1] - t_steps[index]
            source = advance(source, source_v, source_x, dt)
            donor = advance(donor, donor_v, donor_x, dt)
        source_final = source; donor_final = donor
        source_stats = decode_state(source_final, model, config, args.self_cond_cfg)
        donor_stats = decode_state(donor_final, model, config, args.self_cond_cfg)
        source_tokens = torch.tensor([token_map[int(x)] for x in source_batch["target"]], device=device)
        donor_tokens = torch.tensor([token_map[int(x)] for x in donor_batch["target"]], device=device)
        edit_steps = list(map(int, originals[cursor:cursor + bsz]["intervention_step"]))
        positions = source_states[0]["positions"]; batch_rows = torch.arange(bsz, device=device)
        source_endpoint = source_final["z"][batch_rows, positions].float()
        donor_endpoint = donor_final["z"][batch_rows, positions].float()
        endpoint_direction = donor_endpoint - source_endpoint

        for index in range(args.flow_steps):
            dt = t_steps[index + 1] - t_steps[index]
            source_state = source_states[index]; donor_state = donor_states[index]
            source_v_ans = source_velocities[index][batch_rows, positions].float()
            donor_v_ans = donor_velocities[index][batch_rows, positions].float()
            source_next_ans = (source_state["z"] + dt * source_velocities[index])[batch_rows, positions].float()
            donor_next_ans = (donor_state["z"] + dt * donor_velocities[index])[batch_rows, positions].float()
            for component in ("k", "v"):
                patched_v, patched_x = patched_velocity(
                    model, source_state, donor_qkvs[index], component, t_steps[index],
                    config, args.cfg, args.self_cond_cfg, args.block, args.head)
                patched_next = advance(source_state, patched_v, patched_x, dt)
                patched_next_ans = patched_next["z"][batch_rows, positions].float()
                patched_final = finish(model, patched_next, index + 1, t_steps, config,
                                       args.cfg, args.self_cond_cfg)
                patched_stats = decode_state(patched_final, model, config, args.self_cond_cfg)
                delta_v = patched_v[batch_rows, positions].float() - source_v_ans
                endpoint_norm = endpoint_direction.norm(dim=-1).clamp_min(1e-8)
                native_distance = (source_next_ans - donor_next_ans).norm(dim=-1).clamp_min(1e-8)
                patched_distance = (patched_next_ans - donor_next_ans).norm(dim=-1)
                progress = (native_distance - patched_distance) / native_distance
                projection = (dt.float() * delta_v * endpoint_direction).sum(dim=-1) / endpoint_norm.square()
                for local in range(bsz):
                    pred = patched_stats["prediction"][local]
                    edit_step = edit_steps[local]
                    rows.append({
                        "pair_index": cursor + local, "intervention_step": edit_step,
                        "component": component, "time_index": index,
                        "flow_time": float(t_steps[index]),
                        "velocity_shift_norm": float(delta_v[local].norm()),
                        "velocity_endpoint_alignment": float(cosine(delta_v[local:local+1], endpoint_direction[local:local+1])[0]),
                        "velocity_local_donor_alignment": float(cosine(delta_v[local:local+1], (donor_v_ans-source_v_ans)[local:local+1])[0]),
                        "one_step_donor_progress": float(progress[local]),
                        "endpoint_projection": float(projection[local]),
                        "follows_original": bool(pred == source_tokens[local]),
                        "follows_counterfactual": bool(pred == donor_tokens[local]),
                        "follows_other": bool(pred != source_tokens[local] and pred != donor_tokens[local]),
                        "source_baseline_correct": bool(source_stats["correct"][local]),
                        "donor_baseline_correct": bool(donor_stats["correct"][local]),
                        "eligible": bool(source_stats["correct"][local] and donor_stats["correct"][local]),
                        "mechanistically_relevant": bool((component == "k" and edit_step in (1,2,3)) or
                                                         (component == "v" and edit_step == 4)),
                    })
        cursor += bsz
        print(f"Velocity transport: processed {cursor}/{len(originals)} pairs", flush=True)

    metrics = aggregate(rows)
    relevant = [r for r in rows if r["eligible"] and r["mechanistically_relevant"]]
    summary = {"checkpoint": args.checkpoint, "checkpoint_step": checkpoint_step,
        "seed": args.seed, "counts": counts, "pairs": len(originals), "flow_steps": args.flow_steps,
        "t_steps": [float(x) for x in t_steps], "block": args.block, "head": args.head,
        "mechanistic_subsets": {"k": "edits s1-s3", "v": "edit s4"},
        "eligible_relevant_rows": len(relevant), "metrics": metrics}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "transport_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    with (output_dir / "per_sample.jsonl").open("w") as handle:
        for row in rows: handle.write(json.dumps(row) + "\n")
    plot(rows, output_dir, args.flow_steps, t_steps)
    print(f"Saved velocity-transport artifacts to {output_dir}", flush=True)


if __name__ == "__main__": run(parse_args())
