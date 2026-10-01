#!/usr/bin/env python
"""Verifier-backed held-out evaluation for the official ELF Countdown model."""

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "official-elf" / "src"))

from configs.config import SamplingConfig, load_config_from_yaml
from countdown_task import evaluate_rpn
from modules.t5_encoder import get_encoder
from official_diagnostics import load_model
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split
from utils.encoder_utils import encode_text
from utils.generation_utils import _dlm_decode_batch, _generate_samples_single_batch
from utils.sampling_utils import get_sampling_steps


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=1400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--cfgs", nargs="+", type=float, default=[1, 2])
    parser.add_argument("--free-decode-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260928)
    return parser.parse_args()


def verify_expression(text, numbers, target):
    try:
        return evaluate_rpn(text.strip(), list(map(int, numbers))) == int(target)
    except (ValueError, TypeError, IndexError):
        return False


def find_valid_prefix(text, numbers, target):
    tokens = text.strip().split()
    for stop in range(1, len(tokens) + 1):
        candidate = " ".join(tokens[:stop])
        if verify_expression(candidate, numbers, target):
            return candidate
    return None


@torch.no_grad()
def evaluate(model, encoder, dataset, tokenizer, config, device, steps, cfg,
             num_samples, batch_size, seed, free_decode_tokens):
    loader = get_dataloader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        drop_last=False, max_seq_length=config.max_length,
        pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
        max_input_seq_length=config.max_input_length, distributed=False,
    )
    sampling = SamplingConfig(
        sampling_method="ode", num_sampling_steps=[steps], cfgs=[cfg],
        self_cond_cfg_scales=[1.0], time_schedule="logit_normal", sde_gamma=0.0,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dtype = next(model.parameters()).dtype
    exact = valid = free_valid = seen = 0
    grouped = defaultdict(
        lambda: {"samples": 0, "exact": 0, "valid": 0, "free_valid": 0}
    )
    examples = []

    for batch in loader:
        if seen >= num_samples:
            break
        take = min(len(batch["target"]), num_samples - seen)
        input_ids = torch.from_numpy(np.asarray(batch["input_ids"][:take])).to(device).long()
        encoder_mask = torch.from_numpy(
            np.asarray(batch["encoder_attention_mask"][:take])
        ).to(device).float()
        attention_mask = torch.from_numpy(
            np.asarray(batch["attention_mask"][:take])
        ).to(device).float()
        cond_mask = torch.from_numpy(
            np.asarray(batch["cond_seq_mask"][:take])
        ).to(device).float()
        starts = cond_mask.long().sum(dim=1)
        target_mask = (attention_mask * (1.0 - cond_mask)).bool()
        lengths = target_mask.long().sum(dim=1)
        cond_seq = encode_text(
            input_ids, encoder_mask, encoder, config.latent_mean,
            config.latent_std, use_bf16=bool(config.use_bf16),
        ).to(dtype)
        z = (
            torch.randn(
                (take, config.max_length, model.text_encoder_dim),
                generator=generator, dtype=dtype,
            ) * config.denoiser_noise_scale
        ).to(device)
        t_steps = get_sampling_steps(
            steps, "logit_normal", config.denoiser_p_mean,
            config.denoiser_p_std, device=device, dtype=dtype,
        )
        latent = _generate_samples_single_batch(
            model=model, generator=generator, z=z, t_steps=t_steps,
            cond_seq=cond_seq, cond_seq_mask=cond_mask, config=config,
            sampling_config=sampling, cfg_scale=cfg, self_cond_cfg_scale=1.0,
        )
        predicted = _dlm_decode_batch(latent, model, 1.0, config, 1.0)
        metadata = dataset[seen:seen + take]
        for row in range(take):
            start = int(starts[row])
            length = int(lengths[row])
            predicted_ids = predicted[row, start:start + length]
            target_ids = input_ids[row, start:start + length]
            is_exact = bool(predicted_ids.eq(target_ids).all())
            text = tokenizer.decode(
                predicted_ids.detach().cpu().tolist(), skip_special_tokens=True
            ).strip()
            numeric_target = int(str(metadata["key"][row]).rsplit(":", 1)[1])
            is_valid = verify_expression(
                text, metadata["numbers"][row], numeric_target
            )
            free_ids = predicted[
                row, start:min(start + free_decode_tokens, predicted.shape[1])
            ]
            free_text = tokenizer.decode(
                free_ids.detach().cpu().tolist(), skip_special_tokens=True
            ).strip()
            valid_prefix = find_valid_prefix(
                free_text, metadata["numbers"][row], numeric_target
            )
            is_free_valid = valid_prefix is not None
            depth = int(metadata["depth"][row])
            group = grouped[f"d{depth}"]
            group["samples"] += 1
            group["exact"] += int(is_exact)
            group["valid"] += int(is_valid)
            group["free_valid"] += int(is_free_valid)
            exact += int(is_exact)
            valid += int(is_valid)
            free_valid += int(is_free_valid)
            if len(examples) < 30:
                examples.append({
                    "depth": depth,
                    "numbers": list(map(int, metadata["numbers"][row])),
                    "numeric_target": numeric_target,
                    "reference": metadata["target_text"][row],
                    "prediction": text,
                    "free_prediction": free_text,
                    "valid_prefix": valid_prefix,
                    "exact": is_exact,
                    "valid": is_valid,
                    "free_valid": is_free_valid,
                })
        seen += take

    return {
        "steps": steps,
        "cfg": cfg,
        "samples": seen,
        "exact_match": exact / max(seen, 1),
        "verifier_accuracy": valid / max(seen, 1),
        "free_prefix_verifier_accuracy": free_valid / max(seen, 1),
        "groups": {
            name: {
                "samples": values["samples"],
                "exact_match": values["exact"] / max(values["samples"], 1),
                "verifier_accuracy": values["valid"] / max(values["samples"], 1),
                "free_prefix_verifier_accuracy": (
                    values["free_valid"] / max(values["samples"], 1)
                ),
            }
            for name, values in sorted(grouped.items())
        },
        "examples": examples,
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config_from_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name or config.encoder_model_name
    )
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    model, checkpoint_step = load_model(
        config, args.checkpoint, encoder_config, tokenizer, device
    )
    dataset = load_dataset_split(config.eval_data_path)
    report = {"checkpoint_step": checkpoint_step, "results": []}
    for steps, cfg in itertools.product(args.steps, args.cfgs):
        print(f"Evaluating steps={steps}, cfg={cfg}", flush=True)
        report["results"].append(evaluate(
            model, encoder, dataset, tokenizer, config, device, steps, cfg,
            args.num_samples, args.batch_size, args.seed, args.free_decode_tokens,
        ))
        print(json.dumps(report["results"][-1]), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
