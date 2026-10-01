#!/usr/bin/env python
"""Measure how a localized reasoning intervention propagates through ELF flow.

For each source/donor counterfactual pair, this script applies a one-step pulse
to a previously localized block-11/head-3 semantic K or V component at flow
time s.  It then resumes the unmodified source-conditioned dynamics and records
the response at every later observation time t.  The output is a causal
response kernel over intervention time x observation time.
"""

import argparse
import csv
import json
import math
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
from official_counterfactual_patching import build_pairs_for_depth
from official_diagnostics import load_model
from official_flow_layer_causal_map import prepare_flow
from official_head_position_patching import position_mappings
from official_layerwise_patching import semantic_token_map
from official_reasoning_velocity_transport import advance, forward_velocity
from official_semantic_velocity_transport import MODES, semantic_patched_velocity
from official_temporal_causal_tracing import capture_qkv
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split
from utils.generation_utils import _dlm_decode_logits
from utils.sampling_utils import get_sampling_steps


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pairs-per-step", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--self-cond-cfg", type=float, default=1.0)
    parser.add_argument("--block", type=int, default=11)
    parser.add_argument("--head", type=int, default=3)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--mode", action="append", choices=tuple(MODES), default=None)
    parser.add_argument("--strength", action="append", type=float, default=None)
    parser.add_argument("--time-index", action="append", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260928)
    return parser.parse_args()


def clone_state(state):
    result = dict(state)
    result["z"] = state["z"].clone()
    result["previous"] = state["previous"].clone()
    return result


def answer_logit_contrast(state, model, config, self_cond_cfg, source_tokens, donor_tokens):
    logits = _dlm_decode_logits(state["z"], model, 1.0, config, self_cond_cfg)
    rows = torch.arange(state["z"].shape[0], device=state["z"].device)
    answer_logits = logits[rows, state["positions"]]
    source_logits = answer_logits.gather(1, source_tokens[:, None]).squeeze(1)
    donor_logits = answer_logits.gather(1, donor_tokens[:, None]).squeeze(1)
    return (donor_logits - source_logits).float(), answer_logits.argmax(dim=-1)


def safe_cosine(a, b):
    return F.cosine_similarity(a.float(), b.float(), dim=-1, eps=1e-8)


def is_relevant_edit(mode, edit_step, depth):
    if mode.startswith("k_"):
        return edit_step < depth
    if mode.startswith("v_"):
        return edit_step == depth
    raise ValueError(f"Unknown intervention mode: {mode}")


def aggregate(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["mode"], row["strength"], row["intervention_index"],
                 row["observation_index"])].append(row)
    metrics = []
    fields = (
        "answer_axis_projection", "answer_response_norm", "global_response_norm",
        "response_gain", "response_persistence", "semantic_ftle",
        "logit_contrast_response", "donor_answer_rate", "source_answer_rate",
    )
    for key, items in sorted(buckets.items()):
        relevant = [item for item in items if item["eligible"] and item["relevant_edit"]]
        chosen = relevant or [item for item in items if item["eligible"]] or items
        metric = {
            "mode": key[0], "strength": key[1],
            "intervention_index": key[2], "observation_index": key[3],
            "intervention_time": items[0]["intervention_time"],
            "observation_time": items[0]["observation_time"],
            "samples": len(items), "relevant_samples": len(relevant),
        }
        for field in fields:
            metric[field] = float(np.mean([item[field] for item in chosen]))
        metrics.append(metric)
    return metrics


def matrix(metrics, mode, strength, field, flow_steps):
    # rows: observation state index 1..N; columns: pulse index 0..N-1
    values = np.full((flow_steps, flow_steps), np.nan)
    for row in metrics:
        if row["mode"] == mode and row["strength"] == strength:
            values[row["observation_index"] - 1, row["intervention_index"]] = row[field]
    return values


def plot(metrics, output_dir, modes, strengths, t_steps, flow_steps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = (
        ("answer_axis_projection", "Response along donor-answer axis", "coolwarm", None),
        ("response_gain", "Response amplification", "magma", 0.0),
        ("logit_contrast_response", "Donor-vs-source logit response", "coolwarm", None),
    )
    for strength in strengths:
        fig, axes = plt.subplots(len(modes), len(fields),
                                 figsize=(5.4 * len(fields), 4.6 * len(modes)),
                                 squeeze=False)
        for row_index, mode in enumerate(modes):
            for col_index, (field, title, cmap, vmin) in enumerate(fields):
                axis = axes[row_index, col_index]
                values = matrix(metrics, mode, strength, field, flow_steps)
                finite = values[np.isfinite(values)]
                if field == "response_gain":
                    vmax = max(1.0, float(np.quantile(finite, .95))) if finite.size else 1.0
                else:
                    vmax = float(np.quantile(np.abs(finite), .95)) if finite.size else 1.0
                    vmax = max(vmax, 1e-6)
                    vmin = -vmax
                image = axis.imshow(values, origin="lower", aspect="auto", cmap=cmap,
                                    vmin=vmin, vmax=vmax)
                axis.set_title(f"{mode}\n{title}")
                axis.set_xlabel("Intervention time s")
                axis.set_ylabel("Observation time t")
                axis.set_xticks(range(flow_steps),
                                [f"{float(t_steps[i]):.2f}" for i in range(flow_steps)],
                                rotation=35)
                axis.set_yticks(range(flow_steps),
                                [f"{float(t_steps[i + 1]):.2f}" for i in range(flow_steps)])
                fig.colorbar(image, ax=axis, fraction=.046, pad=.04)
        fig.suptitle(f"Causal response kernel (patch strength={strength:g})", y=.995)
        fig.tight_layout()
        label = str(strength).replace(".", "p")
        fig.savefig(output_dir / f"causal_response_kernel_strength_{label}.png", dpi=220)
        plt.close(fig)

    intervention_indices = sorted(set(row["intervention_index"] for row in metrics))
    fig, axes = plt.subplots(len(modes), 3, figsize=(15, 4.1 * len(modes)), squeeze=False)
    dose_fields = (
        ("answer_axis_projection", "Immediate answer-axis response", "first"),
        ("answer_axis_projection", "Final answer-axis response", "last"),
        ("donor_answer_rate", "Final donor-answer rate", "last"),
    )
    for row_index, mode in enumerate(modes):
        for col_index, (field, title, location) in enumerate(dose_fields):
            axis = axes[row_index, col_index]
            for intervention_index in intervention_indices:
                values = []
                for strength in strengths:
                    chosen = [row for row in metrics
                              if row["mode"] == mode and row["strength"] == strength
                              and row["intervention_index"] == intervention_index]
                    chosen = sorted(chosen, key=lambda row: row["observation_index"])
                    values.append((chosen[0] if location == "first" else chosen[-1])[field])
                axis.plot(strengths, values, marker="o",
                          label=f"s={float(t_steps[intervention_index]):.3g}")
            axis.set_title(f"{mode}\n{title}")
            axis.set_xlabel("Patch strength")
            axis.grid(alpha=.25)
            if field == "donor_answer_rate":
                axis.set_ylim(-.03, 1.03)
            if col_index == 0:
                axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "causal_dose_response.png", dpi=220)
    plt.close(fig)


@torch.no_grad()
def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = args.mode or [
        "k_final_table", "k_edited_table_control",
        "v_changed_cell", "v_swap_partner_control",
    ]
    strengths = args.strength or [0.25, 1.0]
    intervention_indices = args.time_index or list(range(args.flow_steps))
    if any(strength <= 0 for strength in strengths):
        raise ValueError("Patch strengths must be positive")
    if any(index < 0 or index >= args.flow_steps for index in intervention_indices):
        raise ValueError("Time indices must lie in [0, flow_steps)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config_from_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    token_map = semantic_token_map(tokenizer)
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    model, checkpoint_step = load_model(config, args.checkpoint, encoder_config, tokenizer, device)
    validation = load_dataset_split(config.eval_data_path)
    originals, counterfactuals, counts = build_pairs_for_depth(
        validation, tokenizer, args.pairs_per_step, args.depth, args.seed)
    loader_args = dict(
        batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False,
        max_seq_length=config.max_length,
        pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
        max_input_seq_length=config.max_input_length, distributed=False,
    )
    source_loader = get_dataloader(originals, **loader_args)
    donor_loader = get_dataloader(counterfactuals, **loader_args)
    dtype = next(model.parameters()).dtype
    t_steps = get_sampling_steps(
        args.flow_steps, "uniform", config.denoiser_p_mean,
        config.denoiser_p_std, device=device, dtype=dtype)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    use_bf16 = bool(config.use_bf16) and device.type == "cuda"
    rows = []
    cursor = 0

    for source_batch, donor_batch in zip(source_loader, donor_loader):
        bsz = len(source_batch["target"])
        noise = torch.randn((bsz, config.max_length, model.text_encoder_dim),
                            generator=generator, dtype=dtype)
        source = prepare_flow(source_batch, noise, model, encoder, tokenizer, config, device)
        donor = prepare_flow(donor_batch, noise, model, encoder, tokenizer, config, device)
        source_rows = [originals[cursor + i] for i in range(bsz)]
        donor_rows = [counterfactuals[cursor + i] for i in range(bsz)]
        mapping_sets = [
            position_mappings(source_rows[i], donor_rows[i], source["positions"][i],
                              donor["positions"][i], tokenizer)
            for i in range(bsz)
        ]
        edit_steps = [int(row["intervention_step"]) for row in source_rows]
        source_tokens = torch.tensor(
            [token_map[int(x)] for x in source_batch["target"]], device=device)
        donor_tokens = torch.tensor(
            [token_map[int(x)] for x in donor_batch["target"]], device=device)
        batch_rows = torch.arange(bsz, device=device)
        positions = source["positions"]

        source_states = [clone_state(source)]
        donor_states = [clone_state(donor)]
        donor_qkvs = []
        for index in range(args.flow_steps):
            donor_qkvs.append(capture_qkv(
                model, donor, t_steps[index], args.block, args.self_cond_cfg, use_bf16))
            source_v, source_x = forward_velocity(
                model, source, t_steps[index], config, args.cfg, args.self_cond_cfg)
            donor_v, donor_x = forward_velocity(
                model, donor, t_steps[index], config, args.cfg, args.self_cond_cfg)
            dt = t_steps[index + 1] - t_steps[index]
            source = advance(source, source_v, source_x, dt)
            donor = advance(donor, donor_v, donor_x, dt)
            source_states.append(clone_state(source))
            donor_states.append(clone_state(donor))

        source_contrasts = []
        for state in source_states:
            contrast, _ = answer_logit_contrast(
                state, model, config, args.self_cond_cfg, source_tokens, donor_tokens)
            source_contrasts.append(contrast)
        _, source_final_predictions = answer_logit_contrast(
            source_states[-1], model, config, args.self_cond_cfg, source_tokens, donor_tokens)
        _, donor_final_predictions = answer_logit_contrast(
            donor_states[-1], model, config, args.self_cond_cfg, source_tokens, donor_tokens)
        eligible = ((source_final_predictions == source_tokens) &
                    (donor_final_predictions == donor_tokens))

        for intervention_index in intervention_indices:
            state = source_states[intervention_index]
            dt = t_steps[intervention_index + 1] - t_steps[intervention_index]
            for mode in modes:
                component, mapping_name, _ = MODES[mode]
                mappings = [item[mapping_name] for item in mapping_sets]
                for strength in strengths:
                    patched_v, patched_x = semantic_patched_velocity(
                        model, state, donor_qkvs[intervention_index], component, mappings,
                        t_steps[intervention_index], config, args.cfg,
                        args.self_cond_cfg, args.block, args.head, strength=strength)
                    branch = advance(state, patched_v, patched_x, dt)
                    initial_delta = (
                        branch["z"][batch_rows, positions].float()
                        - source_states[intervention_index + 1]["z"][batch_rows, positions].float()
                    )
                    initial_norm = initial_delta.norm(dim=-1).clamp_min(1e-8)

                    for observation_index in range(intervention_index + 1, args.flow_steps + 1):
                        source_obs = source_states[observation_index]
                        donor_obs = donor_states[observation_index]
                        response = (branch["z"][batch_rows, positions].float()
                                    - source_obs["z"][batch_rows, positions].float())
                        donor_axis = (donor_obs["z"][batch_rows, positions].float()
                                      - source_obs["z"][batch_rows, positions].float())
                        axis_norm_sq = donor_axis.square().sum(dim=-1).clamp_min(1e-8)
                        response_norm = response.norm(dim=-1)
                        projection = (response * donor_axis).sum(dim=-1) / axis_norm_sq
                        global_response = (branch["z"].float() - source_obs["z"].float())
                        global_norm = global_response.flatten(1).norm(dim=-1)
                        gain = response_norm / initial_norm
                        persistence = safe_cosine(response, initial_delta)
                        elapsed = float(t_steps[observation_index] - t_steps[intervention_index + 1])
                        if elapsed > 1e-8:
                            ftle = torch.log(gain.clamp_min(1e-8)) / elapsed
                        else:
                            ftle = torch.zeros_like(gain)
                        contrast, prediction = answer_logit_contrast(
                            branch, model, config, args.self_cond_cfg, source_tokens, donor_tokens)
                        contrast_response = contrast - source_contrasts[observation_index]

                        for local in range(bsz):
                            rows.append({
                                "pair_index": cursor + local,
                                "intervention_step": edit_steps[local],
                                "mode": mode, "component": component,
                                "strength": float(strength),
                                "intervention_index": intervention_index,
                                "observation_index": observation_index,
                                "intervention_time": float(t_steps[intervention_index]),
                                "observation_time": float(t_steps[observation_index]),
                                "relevant_edit": is_relevant_edit(
                                    mode, edit_steps[local], args.depth),
                                "eligible": bool(eligible[local]),
                                "answer_axis_projection": float(projection[local]),
                                "answer_response_norm": float(response_norm[local]),
                                "global_response_norm": float(global_norm[local]),
                                "response_gain": float(gain[local]),
                                "response_persistence": float(persistence[local]),
                                "semantic_ftle": float(ftle[local]),
                                "logit_contrast_response": float(contrast_response[local]),
                                "donor_answer_rate": bool(prediction[local] == donor_tokens[local]),
                                "source_answer_rate": bool(prediction[local] == source_tokens[local]),
                            })

                        if observation_index < args.flow_steps:
                            branch_v, branch_x = forward_velocity(
                                model, branch, t_steps[observation_index], config,
                                args.cfg, args.self_cond_cfg)
                            next_dt = t_steps[observation_index + 1] - t_steps[observation_index]
                            branch = advance(branch, branch_v, branch_x, next_dt)

        cursor += bsz
        print(f"Causal response kernel: processed {cursor}/{len(originals)}", flush=True)

    metrics = aggregate(rows)
    summary = {
        "checkpoint": args.checkpoint, "checkpoint_step": checkpoint_step,
        "seed": args.seed, "depth": args.depth,
        "counts": counts, "pairs": len(originals),
        "flow_steps": args.flow_steps, "t_steps": [float(x) for x in t_steps],
        "modes": modes, "strengths": strengths,
        "intervention_indices": intervention_indices,
        "kernel_definition": (
            "single-flow-step semantic QKV pulse at s; native source-conditioned "
            "dynamics thereafter; response observed at every t>s"),
        "metrics": metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "response_kernel_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    with (output_dir / "per_sample.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    plot(metrics, output_dir, modes, strengths, t_steps, args.flow_steps)
    print(f"Saved causal response kernel to {output_dir}", flush=True)


if __name__ == "__main__":
    run(parse_args())
