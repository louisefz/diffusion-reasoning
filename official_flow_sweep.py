#!/usr/bin/env python
"""Evaluate flow sampling while ignoring untrained padding positions."""

import argparse
from collections import defaultdict
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
OFFICIAL_SRC = ROOT / "official-elf" / "src"
sys.path.insert(0, str(OFFICIAL_SRC))

from configs.config import SamplingConfig, load_config_from_yaml
from modules.t5_encoder import get_encoder
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split
from utils.encoder_utils import encode_text
from utils.generation_utils import _dlm_decode_batch, _generate_samples_single_batch
from utils.sampling_utils import get_sampling_steps
from official_diagnostics import load_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--schedules", nargs="+", default=["uniform", "logit_normal"])
    parser.add_argument("--steps", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--cfgs", nargs="+", type=float, default=[1, 2, 3])
    parser.add_argument("--reasoning-loops", nargs="+", type=int, default=[1])
    parser.add_argument("--reasoning-loop-scale", type=float, default=1.0)
    parser.add_argument("--reasoning-memory-tokens", type=int, default=0)
    parser.add_argument("--no-reasoning-inner-time", action="store_true")
    parser.add_argument("--reasoning-memory-direct-coupling", action="store_true")
    parser.add_argument("--reasoning-memory-bottleneck", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


@torch.no_grad()
def evaluate_setting(model, encoder, dataset, tokenizer, config, setting, device,
                     num_samples, batch_size):
    loader = get_dataloader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False,
        max_seq_length=config.max_length,
        pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
        max_input_seq_length=config.max_input_length, distributed=False,
    )
    sampling = SamplingConfig(
        sampling_method="ode", num_sampling_steps=[setting["steps"]],
        cfgs=[setting["cfg"]], self_cond_cfg_scales=[1.0],
        time_schedule=setting["schedule"], sde_gamma=0.0,
    )
    generator = torch.Generator(device="cpu").manual_seed(20260908)
    torch.manual_seed(20260908)
    torch.cuda.manual_seed_all(20260908)
    token_correct = token_total = first_correct = exact_correct = seen = 0
    first_target_predictions = []
    exact_correct_flags = []
    grouped = defaultdict(lambda: {
        "samples": 0, "target_token_correct": 0, "target_tokens": 0,
        "first_correct": 0, "exact_correct": 0,
    })
    dtype = next(model.parameters()).dtype

    for batch in loader:
        if seen >= num_samples:
            break
        take = min(len(batch["target"]), num_samples - seen)
        input_ids = torch.from_numpy(np.asarray(batch["input_ids"][:take])).to(device).long()
        encoder_mask = torch.from_numpy(np.asarray(batch["encoder_attention_mask"][:take])).to(device).float()
        attention_mask = torch.from_numpy(np.asarray(batch["attention_mask"][:take])).to(device).float()
        cond_mask = torch.from_numpy(np.asarray(batch["cond_seq_mask"][:take])).to(device).float()
        cond_lengths = cond_mask.to(torch.long).sum(dim=1)
        target_mask = (attention_mask * (1.0 - cond_mask)).bool()
        cond_seq = encode_text(
            input_ids, encoder_mask, encoder, config.latent_mean, config.latent_std,
            use_bf16=bool(config.use_bf16),
        ).to(dtype)
        z = (torch.randn((take, config.max_length, model.text_encoder_dim),
                         generator=generator, dtype=dtype) * config.denoiser_noise_scale).to(device)
        t_steps = get_sampling_steps(
            setting["steps"], setting["schedule"], config.denoiser_p_mean,
            config.denoiser_p_std, device=device, dtype=dtype,
        )
        latent = _generate_samples_single_batch(
            model=model, generator=generator, z=z, t_steps=t_steps,
            cond_seq=cond_seq, cond_seq_mask=cond_mask, config=config,
            sampling_config=sampling, cfg_scale=setting["cfg"], self_cond_cfg_scale=1.0,
        )
        predicted = _dlm_decode_batch(latent, model, 1.0, config, 1.0)
        matches = predicted == input_ids
        row_token_correct = (matches & target_mask).sum(dim=1)
        row_token_total = target_mask.sum(dim=1)
        token_correct += row_token_correct.sum().item()
        token_total += row_token_total.sum().item()
        rows = torch.arange(take, device=device)
        row_first_correct = matches[rows, cond_lengths]
        row_exact_correct = (matches | ~target_mask).all(dim=1)
        first_target_predictions.extend(
            predicted[rows, cond_lengths].detach().cpu().tolist())
        exact_correct_flags.extend(row_exact_correct.detach().cpu().tolist())
        first_correct += row_first_correct.sum().item()
        exact_correct += row_exact_correct.sum().item()
        if "task" in dataset.column_names and "depth" in dataset.column_names:
            metadata = dataset[seen:seen + take]
            token_correct_values = row_token_correct.cpu().tolist()
            token_total_values = row_token_total.cpu().tolist()
            first_values = row_first_correct.cpu().tolist()
            exact_values = row_exact_correct.cpu().tolist()
            for index, (task, depth) in enumerate(zip(metadata["task"], metadata["depth"])):
                group = grouped[f"{task}/d{depth}"]
                group["samples"] += 1
                group["target_token_correct"] += int(token_correct_values[index])
                group["target_tokens"] += int(token_total_values[index])
                group["first_correct"] += int(first_values[index])
                group["exact_correct"] += int(exact_values[index])
        seen += take
    result = {
        **setting,
        "samples": seen,
        "target_token_accuracy": token_correct / max(token_total, 1),
        "first_target_token_accuracy": first_correct / max(seen, 1),
        "all_valid_target_tokens_accuracy": exact_correct / max(seen, 1),
        # These compact per-example traces distinguish genuinely identical
        # predictions from equal aggregate accuracies across loop counts.
        "first_target_predictions": first_target_predictions,
        "exact_correct_flags": exact_correct_flags,
    }
    if grouped:
        result["groups"] = {
            name: {
                "samples": values["samples"],
                "target_token_accuracy": values["target_token_correct"] / max(values["target_tokens"], 1),
                "first_target_token_accuracy": values["first_correct"] / max(values["samples"], 1),
                "all_valid_target_tokens_accuracy": values["exact_correct"] / max(values["samples"], 1),
            }
            for name, values in sorted(grouped.items())
        }
    return result


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config_from_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    dataset = load_dataset_split(config.eval_data_path)
    settings = [
        {"schedule": schedule, "steps": steps, "cfg": float(cfg)}
        for schedule, steps, cfg in itertools.product(
            args.schedules, args.steps, args.cfgs
        )
    ]
    report = {"chance_accuracy": 0.125, "results": []}
    for checkpoint_spec in args.checkpoint:
        name, path = checkpoint_spec.split("=", 1)
        for loops in args.reasoning_loops:
            config.reasoning_loops = loops
            config.reasoning_loop_scale = args.reasoning_loop_scale
            config.reasoning_memory_tokens = args.reasoning_memory_tokens
            config.reasoning_memory_inner_time = not args.no_reasoning_inner_time
            config.reasoning_memory_direct_coupling = args.reasoning_memory_direct_coupling
            config.reasoning_memory_bottleneck = args.reasoning_memory_bottleneck
            model, checkpoint_step = load_model(
                config, path, encoder_config, tokenizer, device)
            print(
                "recurrent_config:",
                {
                    "loops": model.reasoning_loops,
                    "start": model.reasoning_loop_start,
                    "end": model.reasoning_loop_end,
                    "scale": model.reasoning_loop_scale,
                },
                flush=True,
            )
            loop_name = f"{name}/K{loops}"
            for setting in settings:
                print(f"{loop_name}: {setting}", flush=True)
                metrics = evaluate_setting(
                    model, encoder, dataset, tokenizer, config, setting, device,
                    args.num_samples, args.batch_size,
                )
                metrics.update({
                    "name": loop_name, "checkpoint_step": checkpoint_step,
                    "reasoning_loops": loops,
                    "reasoning_loop_scale": args.reasoning_loop_scale,
                })
                report["results"].append(metrics)
                print(json.dumps(metrics), flush=True)
            del model
            torch.cuda.empty_cache()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
