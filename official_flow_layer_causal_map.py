#!/usr/bin/env python
"""Two-dimensional causal patching over ELF flow time and denoiser depth.

For minimal counterfactual composition pairs, inject the counterfactual
answer-slot residual at a chosen (flow step, block), finish the native ODE
trajectory, and measure whether the final decoded answer follows the donor.
"""

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
from official_counterfactual_patching import build_pairs
from official_diagnostics import load_model
from official_flow_trajectory import answer_decoder_stats
from official_layerwise_patching import HiddenRecorder, semantic_token_map
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split
from utils.encoder_utils import encode_text
from utils.generation_utils import _dlm_decode_logits
from utils.sampling_utils import _ode_step, get_sampling_steps, restore_cond


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
    parser.add_argument("--layer", action="append", type=int, default=None)
    parser.add_argument("--time-index", action="append", type=int, default=None)
    parser.add_argument("--mode", action="append", choices=("pulse", "sustained"), default=None)
    parser.add_argument("--seed", type=int, default=20260920)
    return parser.parse_args()


def prepare_flow(batch, shared_noise, model, encoder, tokenizer, config, device):
    dtype = next(model.parameters()).dtype
    input_ids = torch.from_numpy(np.asarray(batch["input_ids"])).to(device).long()
    encoder_mask = torch.from_numpy(
        np.asarray(batch["encoder_attention_mask"])
    ).to(device).float()
    cond_mask = torch.from_numpy(np.asarray(batch["cond_seq_mask"])).to(device).float()
    starts = cond_mask.long().sum(dim=1)
    lengths = torch.tensor(
        [len(tokenizer.encode(str(target), add_special_tokens=False))
         for target in batch["target"]], device=device, dtype=torch.long,
    )
    positions = starts + lengths - 1
    cond_seq = encode_text(
        input_ids, encoder_mask, encoder, config.latent_mean,
        config.latent_std, use_bf16=bool(config.use_bf16),
    ).to(dtype)
    z = shared_noise.to(device=device, dtype=dtype) * config.denoiser_noise_scale
    z = restore_cond(z, cond_seq, cond_mask)
    previous = restore_cond(torch.zeros_like(z), cond_seq, cond_mask)
    return {
        "input_ids": input_ids, "cond_mask": cond_mask, "cond_seq": cond_seq,
        "starts": starts, "lengths": lengths, "positions": positions,
        "z": z, "previous": previous,
    }


def capture_conditional_hidden(model, state, t, self_cond_cfg, use_bf16):
    bsz = state["z"].shape[0]
    dtype = state["z"].dtype
    model_input = torch.cat([state["z"], state["previous"]], dim=-1)
    t_batch = torch.full((bsz,), float(t), device=state["z"].device, dtype=dtype)
    sc_batch = torch.full(
        (bsz,), float(self_cond_cfg), device=state["z"].device, dtype=dtype,
    )
    recorder = HiddenRecorder(model, state["positions"])
    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            model(
                model_input, t_batch, deterministic=True,
                self_cond_cfg_scale=sc_batch, decoder_step_active=None,
            )
    finally:
        recorder.close()
    return torch.stack(recorder.hidden, dim=1)


def ode_step(model, state, t, t_next, config, cfg, self_cond_cfg):
    z, previous = _ode_step(
        model=model, z=state["z"], t=float(t), t_next=float(t_next),
        x_pred_prev=state["previous"], config=config, cfg_scale=cfg,
        self_cond_cfg_scale=self_cond_cfg, cond_seq=state["cond_seq"],
        cond_seq_mask=state["cond_mask"],
    )
    result = dict(state)
    result["z"], result["previous"] = z, previous
    return result


def patched_ode_step(model, state, donor_vector, layer, t, t_next, config,
                     cfg, self_cond_cfg):
    """Patch only the conditional model call inside the CFG ODE step."""
    rows = torch.arange(state["z"].shape[0], device=state["z"].device)
    prefix = (
        model.num_model_mode_tokens + model.num_time_tokens
        + model.num_self_cond_cfg_tokens
    )
    calls = 0

    def hook(_module, _inputs, output):
        nonlocal calls
        current_call = calls
        calls += 1
        # _ode_step calls conditional first and unconditional second when cfg != 1.
        if current_call != 0:
            return output
        patched = output.clone()
        positions = state["positions"] if layer == 0 else state["positions"] + prefix
        patched[rows, positions] = donor_vector.to(patched.dtype)
        return patched

    module = model.text_proj if layer == 0 else model.blocks[layer - 1]
    handle = module.register_forward_hook(hook)
    try:
        result = ode_step(
            model, state, t, t_next, config, cfg, self_cond_cfg,
        )
    finally:
        handle.remove()
    expected_calls = 1 if cfg == 1.0 else 2
    if calls != expected_calls:
        raise RuntimeError(f"Expected {expected_calls} model calls, observed {calls}")
    return result


def decode_state(state, model, config, self_cond_cfg):
    logits = _dlm_decode_logits(state["z"], model, 1.0, config, self_cond_cfg)
    return answer_decoder_stats(
        logits, state["starts"], state["lengths"], state["input_ids"],
    )


def aggregate(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["mode"], row["time_index"], row["layer"], row["intervention_step"])].append(row)
    result = []
    for (mode, time_index, layer, intervention_step), items in sorted(buckets.items()):
        result.append({
            "mode": mode, "time_index": time_index,
            "flow_time": items[0]["flow_time"], "layer": layer,
            "intervention_step": intervention_step, "samples": len(items),
            "counterfactual_answer_rate": float(np.mean([
                item["follows_counterfactual"] for item in items
            ])),
            "original_answer_rate": float(np.mean([
                item["follows_original"] for item in items
            ])),
            "other_answer_rate": float(np.mean([
                item["follows_other"] for item in items
            ])),
            "eligible_rate": float(np.mean([item["eligible"] for item in items])),
        })
    return result


def plot(rows, output_dir, modes, time_indices, layers, t_steps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def matrix(mode, intervention_step=None):
        values = np.full((len(layers), len(time_indices)), np.nan)
        for yi, layer in enumerate(layers):
            for xi, time_index in enumerate(time_indices):
                chosen = [r for r in rows if r["mode"] == mode
                          and r["layer"] == layer and r["time_index"] == time_index]
                if intervention_step is not None:
                    chosen = [r for r in chosen if r["intervention_step"] == intervention_step]
                eligible = [r for r in chosen if r["eligible"]]
                if eligible:
                    values[yi, xi] = np.mean([r["follows_counterfactual"] for r in eligible])
        return values

    fig, axes = plt.subplots(1, len(modes), figsize=(7 * len(modes), 5), squeeze=False)
    for axis, mode in zip(axes[0], modes):
        values = matrix(mode)
        image = axis.imshow(values, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis")
        axis.set_title(f"{mode}: final answer follows donor")
        axis.set_xlabel("Patched flow time $t$")
        axis.set_ylabel("Patched block")
        axis.set_xticks(range(len(time_indices)), [f"{float(t_steps[i]):.2f}" for i in time_indices])
        axis.set_yticks(range(len(layers)), layers)
        fig.colorbar(image, ax=axis, label="Counterfactual answer rate")
    fig.tight_layout()
    fig.savefig(output_dir / "flow_time_x_layer_causal_map.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(len(modes), 4, figsize=(17, 4.3 * len(modes)), squeeze=False)
    for row_index, mode in enumerate(modes):
        for step in range(1, 5):
            axis = axes[row_index, step - 1]
            image = axis.imshow(matrix(mode, step), origin="lower", aspect="auto",
                                vmin=0, vmax=1, cmap="viridis")
            axis.set_title(f"{mode}, edit s{step}")
            axis.set_xlabel("Flow time $t$")
            axis.set_ylabel("Block")
            axis.set_xticks(range(len(time_indices)), [f"{float(t_steps[i]):.2f}" for i in time_indices])
            axis.set_yticks(range(len(layers)), layers)
    fig.colorbar(image, ax=axes, fraction=.015, pad=.02, label="Counterfactual answer rate")
    fig.subplots_adjust(left=.05, right=.92, bottom=.08, top=.93, wspace=.18, hspace=.25)
    fig.savefig(output_dir / "flow_time_x_layer_by_edit_step.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config_from_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name or config.encoder_model_name
    )
    token_map = semantic_token_map(tokenizer)
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    model, checkpoint_step = load_model(
        config, args.checkpoint, encoder_config, tokenizer, device,
    )
    validation = load_dataset_split(config.eval_data_path)
    originals, counterfactuals, counts = build_pairs(
        validation, tokenizer, args.pairs_per_step, args.seed,
    )
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
        config.denoiser_p_std, device=device, dtype=dtype,
    )
    layers = args.layer or [8, 9, 10, 11, 12]
    time_indices = args.time_index or list(range(args.flow_steps))
    modes = args.mode or ["pulse", "sustained"]
    for layer in layers:
        if not 0 <= layer <= model.depth:
            raise ValueError(f"Invalid layer {layer}")
    for time_index in time_indices:
        if not 0 <= time_index < args.flow_steps:
            raise ValueError(f"Invalid time index {time_index}")

    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    use_bf16 = bool(config.use_bf16) and device.type == "cuda"
    rows = []
    cursor = 0
    for source_batch, donor_batch in zip(source_loader, donor_loader):
        bsz = len(source_batch["target"])
        noise = torch.randn(
            (bsz, config.max_length, model.text_encoder_dim),
            generator=generator, dtype=dtype,
        )
        source = prepare_flow(
            source_batch, noise, model, encoder, tokenizer, config, device,
        )
        donor = prepare_flow(
            donor_batch, noise, model, encoder, tokenizer, config, device,
        )
        source_states = [source]
        donor_hidden = []
        for step_index in range(args.flow_steps):
            donor_hidden.append(capture_conditional_hidden(
                model, donor, t_steps[step_index], args.self_cond_cfg, use_bf16,
            ).detach())
            source = ode_step(
                model, source, t_steps[step_index], t_steps[step_index + 1],
                config, args.cfg, args.self_cond_cfg,
            )
            donor = ode_step(
                model, donor, t_steps[step_index], t_steps[step_index + 1],
                config, args.cfg, args.self_cond_cfg,
            )
            source_states.append(source)
        source_baseline = decode_state(source, model, config, args.self_cond_cfg)
        donor_baseline = decode_state(donor, model, config, args.self_cond_cfg)
        source_tokens = torch.tensor(
            [token_map[int(x)] for x in source_batch["target"]], device=device,
        )
        donor_tokens = torch.tensor(
            [token_map[int(x)] for x in donor_batch["target"]], device=device,
        )
        edit_steps = list(map(
            int, originals[cursor:cursor + bsz]["intervention_step"],
        ))

        for mode in modes:
            for start_index in time_indices:
                for layer in layers:
                    state = dict(source_states[start_index])
                    for step_index in range(start_index, args.flow_steps):
                        should_patch = (
                            step_index == start_index or mode == "sustained"
                        )
                        if should_patch:
                            state = patched_ode_step(
                                model, state, donor_hidden[step_index][:, layer],
                                layer, t_steps[step_index], t_steps[step_index + 1],
                                config, args.cfg, args.self_cond_cfg,
                            )
                        else:
                            state = ode_step(
                                model, state, t_steps[step_index], t_steps[step_index + 1],
                                config, args.cfg, args.self_cond_cfg,
                            )
                    stats = decode_state(state, model, config, args.self_cond_cfg)
                    for local in range(bsz):
                        prediction = stats["prediction"][local]
                        follows_source = bool(prediction == source_tokens[local])
                        follows_donor = bool(prediction == donor_tokens[local])
                        eligible = bool(
                            source_baseline["correct"][local]
                            and donor_baseline["correct"][local]
                        )
                        rows.append({
                            "pair_index": cursor + local,
                            "intervention_step": edit_steps[local],
                            "mode": mode, "time_index": start_index,
                            "flow_time": float(t_steps[start_index]),
                            "layer": layer,
                            "source_answer": int(source_batch["target"][local]),
                            "counterfactual_answer": int(donor_batch["target"][local]),
                            "prediction_token_id": int(prediction),
                            "follows_original": follows_source,
                            "follows_counterfactual": follows_donor,
                            "follows_other": not follows_source and not follows_donor,
                            "source_baseline_correct": bool(source_baseline["correct"][local]),
                            "donor_baseline_correct": bool(donor_baseline["correct"][local]),
                            "eligible": eligible,
                        })
        cursor += bsz
        print(f"Causal map: processed {cursor}/{len(originals)} pairs", flush=True)

    metrics = aggregate(rows)
    summary = {
        "checkpoint": args.checkpoint, "checkpoint_step": checkpoint_step,
        "device": str(device), "seed": args.seed, "counts": counts,
        "flow_steps": args.flow_steps, "schedule": "uniform",
        "t_steps": [float(x) for x in t_steps], "time_indices": time_indices,
        "layers": layers, "modes": modes, "cfg": args.cfg,
        "pulse_definition": "patch donor answer-slot residual at one denoiser step, then continue source flow unpatched",
        "sustained_definition": "patch the same layer from the chosen step through every remaining denoiser step",
        "eligibility": "both original and counterfactual native baseline trajectories decode correctly",
        "source_baseline_accuracy": float(np.mean([r["source_baseline_correct"] for r in rows])),
        "donor_baseline_accuracy": float(np.mean([r["donor_baseline_correct"] for r in rows])),
        "aggregate": metrics,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    with (output_dir / "causal_map_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader(); writer.writerows(metrics)
    with (output_dir / "per_sample.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    plot(rows, output_dir, modes, time_indices, layers, t_steps)
    print(json.dumps({
        "source_baseline_accuracy": summary["source_baseline_accuracy"],
        "donor_baseline_accuracy": summary["donor_baseline_accuracy"],
        "eligible_rows": int(sum(r["eligible"] for r in rows)),
        "rows": len(rows),
    }, indent=2), flush=True)
    print(f"Saved 2D causal map to {output_dir}", flush=True)


if __name__ == "__main__":
    run(parse_args())
