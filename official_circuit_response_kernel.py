#!/usr/bin/env python
"""Flow-time response of the causally sufficient circuit for each task depth."""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "official-elf" / "src"))

from configs.config import load_config_from_yaml
from modules.t5_encoder import get_encoder
from official_causal_response_kernel import (
    aggregate, answer_logit_contrast, clone_state, plot, safe_cosine,
)
from official_counterfactual_patching import build_pairs_for_depth
from official_diagnostics import load_model
from official_flow_layer_causal_map import prepare_flow
from official_head_patching import capture_preprojection
from official_layerwise_patching import semantic_token_map
from official_reasoning_velocity_transport import advance, forward_velocity
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split
from utils.sampling_utils import get_sampling_steps


def parse_heads(value):
    heads = tuple(int(item) for item in value.split(",") if item != "")
    if not heads:
        raise argparse.ArgumentTypeError("Expected a comma-separated head list")
    return heads


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--circuit-heads", type=parse_heads, required=True)
    parser.add_argument("--control-heads", type=parse_heads, required=True)
    parser.add_argument("--pairs-per-step", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--self-cond-cfg", type=float, default=1.0)
    parser.add_argument("--strength", action="append", type=float, default=None)
    parser.add_argument("--time-index", action="append", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260930)
    return parser.parse_args()


def capture_flow_heads(model, state, t, block, self_cond_cfg, use_bf16):
    bsz = state["z"].shape[0]
    t_batch = torch.full((bsz,), float(t), device=state["z"].device,
                         dtype=state["z"].dtype)
    sc = torch.full((bsz,), float(self_cond_cfg), device=state["z"].device,
                    dtype=state["z"].dtype)
    _, heads = capture_preprojection(
        model, torch.cat([state["z"], state["previous"]], dim=-1),
        t_batch, sc, state["positions"], block, use_bf16)
    return heads


def circuit_patched_velocity(model, state, donor_heads, heads, strength, t,
                             config, cfg, self_cond_cfg, block):
    attention = model.blocks[block - 1].attn
    module = attention.proj
    prefix = model.num_model_mode_tokens + model.num_time_tokens + model.num_self_cond_cfg_tokens
    head_dim = attention.dim // attention.num_heads
    rows = torch.arange(state["z"].shape[0], device=state["z"].device)
    selected = list(heads)
    calls = 0

    def hook(_module, inputs):
        nonlocal calls
        current = calls
        calls += 1
        if current != 0:
            return None
        output = inputs[0].clone()
        vectors = output[rows, state["positions"] + prefix].reshape(
            output.shape[0], attention.num_heads, head_dim)
        source = vectors[:, selected]
        donor = donor_heads[:, selected].to(vectors.dtype)
        vectors[:, selected] = source + strength * (donor - source)
        output[rows, state["positions"] + prefix] = vectors.reshape(
            output.shape[0], attention.dim)
        return (output,)

    handle = module.register_forward_pre_hook(hook)
    try:
        velocity, prediction = forward_velocity(
            model, state, t, config, cfg, self_cond_cfg)
    finally:
        handle.remove()
    expected = 1 if cfg == 1.0 else 2
    if calls != expected:
        raise RuntimeError(f"Expected {expected} attention calls, observed {calls}")
    return velocity, prediction


@torch.no_grad()
def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    strengths = args.strength or [1.0]
    intervention_indices = args.time_index or list(range(args.flow_steps))
    modes = {
        "causal_circuit": args.circuit_heads,
        "size_matched_control": args.control_heads,
    }
    if len(args.circuit_heads) != len(args.control_heads):
        raise ValueError("Circuit and control groups must have equal size")

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
        max_input_seq_length=config.max_input_length, distributed=False)
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
        source_tokens = torch.tensor(
            [token_map[int(x)] for x in source_batch["target"]], device=device)
        donor_tokens = torch.tensor(
            [token_map[int(x)] for x in donor_batch["target"]], device=device)
        batch_rows = torch.arange(bsz, device=device)
        positions = source["positions"]
        edit_steps = list(map(int, originals[cursor:cursor + bsz]["intervention_step"]))

        source_states = [clone_state(source)]
        donor_states = [clone_state(donor)]
        donor_heads_by_time = []
        for index in range(args.flow_steps):
            donor_heads_by_time.append(capture_flow_heads(
                model, donor, t_steps[index], args.block,
                args.self_cond_cfg, use_bf16))
            source_v, source_x = forward_velocity(
                model, source, t_steps[index], config, args.cfg, args.self_cond_cfg)
            donor_v, donor_x = forward_velocity(
                model, donor, t_steps[index], config, args.cfg, args.self_cond_cfg)
            dt = t_steps[index + 1] - t_steps[index]
            source = advance(source, source_v, source_x, dt)
            donor = advance(donor, donor_v, donor_x, dt)
            source_states.append(clone_state(source))
            donor_states.append(clone_state(donor))

        source_contrasts = [answer_logit_contrast(
            state, model, config, args.self_cond_cfg, source_tokens, donor_tokens)[0]
            for state in source_states]
        source_final_prediction = answer_logit_contrast(
            source_states[-1], model, config, args.self_cond_cfg,
            source_tokens, donor_tokens)[1]
        donor_final_prediction = answer_logit_contrast(
            donor_states[-1], model, config, args.self_cond_cfg,
            source_tokens, donor_tokens)[1]
        eligible = ((source_final_prediction == source_tokens)
                    & (donor_final_prediction == donor_tokens))

        for intervention_index in intervention_indices:
            state = source_states[intervention_index]
            dt = t_steps[intervention_index + 1] - t_steps[intervention_index]
            for mode, heads in modes.items():
                for strength in strengths:
                    patched_v, patched_x = circuit_patched_velocity(
                        model, state, donor_heads_by_time[intervention_index],
                        heads, strength, t_steps[intervention_index], config,
                        args.cfg, args.self_cond_cfg, args.block)
                    branch = advance(state, patched_v, patched_x, dt)
                    initial_delta = (
                        branch["z"][batch_rows, positions].float()
                        - source_states[intervention_index + 1]["z"][batch_rows, positions].float())
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
                        global_norm = (branch["z"].float() - source_obs["z"].float()).flatten(1).norm(dim=-1)
                        gain = response_norm / initial_norm
                        persistence = safe_cosine(response, initial_delta)
                        elapsed = float(t_steps[observation_index] - t_steps[intervention_index + 1])
                        ftle = (torch.log(gain.clamp_min(1e-8)) / elapsed
                                if elapsed > 1e-8 else torch.zeros_like(gain))
                        contrast, prediction = answer_logit_contrast(
                            branch, model, config, args.self_cond_cfg,
                            source_tokens, donor_tokens)
                        contrast_response = contrast - source_contrasts[observation_index]
                        for local in range(bsz):
                            rows.append({
                                "pair_index": cursor + local,
                                "intervention_step": edit_steps[local],
                                "mode": mode, "component": "head_group_output",
                                "strength": float(strength),
                                "intervention_index": intervention_index,
                                "observation_index": observation_index,
                                "intervention_time": float(t_steps[intervention_index]),
                                "observation_time": float(t_steps[observation_index]),
                                "relevant_edit": True, "eligible": bool(eligible[local]),
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
        print(f"Circuit response d{args.depth}: processed {cursor}/{len(originals)}", flush=True)

    metrics = aggregate(rows)
    summary = {
        "checkpoint": args.checkpoint, "checkpoint_step": checkpoint_step,
        "depth": args.depth, "block": args.block,
        "circuit_heads": list(args.circuit_heads),
        "control_heads": list(args.control_heads),
        "counts": counts, "pairs": len(originals), "flow_steps": args.flow_steps,
        "t_steps": [float(x) for x in t_steps], "modes": list(modes),
        "strengths": strengths, "intervention_indices": intervention_indices,
        "metrics": metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "circuit_response_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader(); writer.writerows(metrics)
    with (output_dir / "per_sample.jsonl").open("w") as handle:
        for row in rows: handle.write(json.dumps(row) + "\n")
    plot(metrics, output_dir, list(modes), strengths, t_steps, args.flow_steps)
    print(f"Saved circuit response to {output_dir}", flush=True)


if __name__ == "__main__":
    run(parse_args())
