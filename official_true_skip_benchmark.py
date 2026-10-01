#!/usr/bin/env python
"""Benchmark true conditional execution for the validated late B11-attention skip."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "official-elf" / "src"))
from configs.config import load_config_from_yaml
from modules.t5_encoder import get_encoder
from official_diagnostics import load_model
from official_flow_layer_causal_map import decode_state, ode_step, prepare_flow
from official_layerwise_probe import select_balanced
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split
from utils.sampling_utils import get_sampling_steps


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True); p.add_argument("--samples-per-group", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=20); p.add_argument("--flow-steps", type=int, default=8)
    p.add_argument("--cutoff", type=float, default=.625); p.add_argument("--cfg", type=float, default=2.)
    p.add_argument("--self-cond-cfg", type=float, default=1.); p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--timing-repeats", type=int, default=20)
    return p.parse_args()


def sample(model, initial_state, t_steps, cutoff_index, config, cfg, self_cond_cfg, skip):
    state = dict(initial_state)
    module = model.blocks[10].attn
    original_forward = module.forward
    def zero_forward(x, *args, **kwargs):
        return torch.zeros_like(x)
    try:
        for step in range(len(t_steps) - 1):
            if skip and step == cutoff_index:
                module.forward = zero_forward
            state = ode_step(model, state, t_steps[step], t_steps[step + 1],
                             config, cfg, self_cond_cfg)
    finally:
        module.forward = original_forward
    return state


def timed_sample(*sample_args):
    torch.cuda.synchronize(); start = time.perf_counter()
    state = sample(*sample_args)
    torch.cuda.synchronize(); return state, time.perf_counter() - start


@torch.no_grad()
def run(args):
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    config = load_config_from_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    model, checkpoint_step = load_model(config, args.checkpoint, encoder_config, tokenizer, device)
    dataset = load_dataset_split(config.eval_data_path)
    selected, groups, source_ids = select_balanced(dataset, args.samples_per_group, args.seed)
    loader = get_dataloader(selected, batch_size=args.batch_size, shuffle=False, num_workers=0,
        drop_last=False, max_seq_length=config.max_length,
        pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
        max_input_seq_length=config.max_input_length, distributed=False)
    dtype = next(model.parameters()).dtype
    t_steps = get_sampling_steps(args.flow_steps, "uniform", config.denoiser_p_mean,
        config.denoiser_p_std, device=device, dtype=dtype)
    cutoff_index = int(torch.argmin(torch.abs(t_steps.float() - args.cutoff)))
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    native_times = []; skip_times = []; agreement = {}; total_agree = total = 0
    timing_state = None
    cursor = 0
    for batch_index, batch in enumerate(loader):
        bsz = len(batch["target"])
        noise = torch.randn((bsz, config.max_length, model.text_encoder_dim), generator=generator, dtype=dtype)
        initial = prepare_flow(batch, noise, model, encoder, tokenizer, config, device)
        if timing_state is None: timing_state = initial
        # Alternate order to reduce drift bias.
        if batch_index % 2:
            skipped, ts = timed_sample(model, initial, t_steps, cutoff_index, config, args.cfg, args.self_cond_cfg, True)
            native, tn = timed_sample(model, initial, t_steps, cutoff_index, config, args.cfg, args.self_cond_cfg, False)
        else:
            native, tn = timed_sample(model, initial, t_steps, cutoff_index, config, args.cfg, args.self_cond_cfg, False)
            skipped, ts = timed_sample(model, initial, t_steps, cutoff_index, config, args.cfg, args.self_cond_cfg, True)
        native_times.append(tn); skip_times.append(ts)
        pn = decode_state(native, model, config, args.self_cond_cfg)["prediction"]
        ps = decode_state(skipped, model, config, args.self_cond_cfg)["prediction"]
        for n, s, group in zip(pn, ps, groups[cursor:cursor + bsz]):
            same = bool(n == s); total += 1; total_agree += int(same)
            stat = agreement.setdefault(group, [0, 0]); stat[0] += int(same); stat[1] += 1
        cursor += bsz

    # Repeated paired timing on one fixed batch, after warmup.
    for skip in (False, True): sample(model, timing_state, t_steps, cutoff_index, config, args.cfg, args.self_cond_cfg, skip)
    paired_native = []; paired_skip = []
    for repeat in range(args.timing_repeats):
        order = (False, True) if repeat % 2 == 0 else (True, False)
        for skip in order:
            _, elapsed = timed_sample(model, timing_state, t_steps, cutoff_index, config,
                                      args.cfg, args.self_cond_cfg, skip)
            (paired_skip if skip else paired_native).append(elapsed)
    native_mean = float(np.mean(paired_native)); skip_mean = float(np.mean(paired_skip))
    summary = {
        "checkpoint": args.checkpoint, "checkpoint_step": checkpoint_step,
        "policy": "true skip block-11 attention for all CFG calls from cutoff onward",
        "cutoff": args.cutoff, "cutoff_index": cutoff_index, "flow_steps": args.flow_steps,
        "samples": total, "native_answer_agreement": total_agree / total,
        "agreement_by_group": {g: a / n for g, (a, n) in agreement.items()},
        "dataset_pass_native_seconds_mean": float(np.mean(native_times)),
        "dataset_pass_skip_seconds_mean": float(np.mean(skip_times)),
        "timing_repeats": args.timing_repeats,
        "paired_native_seconds_mean": native_mean,
        "paired_skip_seconds_mean": skip_mean,
        "speedup": native_mean / skip_mean,
        "latency_reduction_fraction": 1. - skip_mean / native_mean,
        "paired_native_seconds": paired_native, "paired_skip_seconds": paired_skip,
        "theoretical_component_call_reduction": (args.flow_steps - cutoff_index) / (args.flow_steps * model.depth * 2),
        "source_ids": source_ids,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("source_ids", "paired_native_seconds", "paired_skip_seconds")}, indent=2), flush=True)


if __name__ == "__main__": run(args_parser())
