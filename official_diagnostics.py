#!/usr/bin/env python
"""Diagnose the decoder/denoiser boundary of trained official ELF checkpoints."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
OFFICIAL_SRC = ROOT / "official-elf" / "src"
sys.path.insert(0, str(OFFICIAL_SRC))

from configs.config import load_config_from_yaml
from generation import answer_accuracy_metrics
from modules.model import ELF_models
from modules.t5_encoder import get_encoder
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split
from utils.encoder_utils import encode_text
from utils.generation_utils import _dlm_decode_batch, mask_after_eos, shift_left
from utils.sampling_utils import _forward_sample, add_noise


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="NAME=PATH; repeat for multiple checkpoints")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_model(config, checkpoint_path, encoder_config, tokenizer, device):
    model = ELF_models[config.model](
        text_encoder_dim=encoder_config.d_model,
        max_length=config.max_length,
        attn_drop=config.attn_dropout,
        proj_drop=config.proj_dropout,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=len(tokenizer),
        num_model_mode_tokens=config.num_model_mode_tokens,
        bottleneck_dim=config.bottleneck_dim,
        gradient_checkpointing=False,
        reasoning_loops=int(getattr(config, "reasoning_loops", 1)),
        reasoning_loop_start=int(getattr(config, "reasoning_loop_start", 10)),
        reasoning_loop_end=int(getattr(config, "reasoning_loop_end", 11)),
        reasoning_loop_scale=float(getattr(config, "reasoning_loop_scale", 1.0)),
        reasoning_loop_randomize=False,
        reasoning_loop_min=1,
        reasoning_loop_max=int(getattr(config, "reasoning_loop_max", 1)),
        reasoning_memory_tokens=int(getattr(config, "reasoning_memory_tokens", 0)),
        reasoning_memory_inner_time=bool(getattr(config, "reasoning_memory_inner_time", True)),
        reasoning_memory_direct_coupling=bool(getattr(config, "reasoning_memory_direct_coupling", False)),
        reasoning_memory_bottleneck=bool(getattr(config, "reasoning_memory_bottleneck", False)),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    weights = checkpoint.get("ema_params1", checkpoint["params"])
    model.load_state_dict(weights, strict=True)
    return model.to(device).eval(), int(checkpoint.get("step", -1))


def decoded_texts(ids, tokenizer):
    return [tokenizer.decode(row.detach().cpu().numpy(), skip_special_tokens=True) for row in ids]


def score_prediction(predicted_ids, input_ids, target_mask, cond_lengths, references,
                     tokenizer, eos_token_id, pad_token_id, generation_length):
    token_correct = ((predicted_ids == input_ids) * target_mask.bool()).sum().item()
    token_total = target_mask.sum().item()
    shifted = shift_left(predicted_ids, cond_lengths, pad_token_id)[:, :generation_length]
    shifted = mask_after_eos(shifted, eos_token_id=eos_token_id, pad_token_id=pad_token_id)
    texts = decoded_texts(shifted, tokenizer)
    answer = answer_accuracy_metrics(texts, references)
    answer["target_token_accuracy"] = token_correct / max(token_total, 1)
    answer["target_tokens"] = int(token_total)
    return answer, texts


@torch.no_grad()
def diagnose_one(name, checkpoint_path, config, encoder, encoder_config, tokenizer,
                 dataset, device, num_samples, batch_size):
    model, step = load_model(config, checkpoint_path, encoder_config, tokenizer, device)
    pad_id = get_pad_token_id(tokenizer, config.pad_token)
    eos_id = tokenizer.eos_token_id
    loader = get_dataloader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False,
        max_seq_length=config.max_length, pad_token_id=pad_id,
        max_input_seq_length=config.max_input_length, distributed=False,
    )
    accumulators = {"oracle_clean": [], **{f"denoise_t{t}": [] for t in (0.25, 0.5, 0.75, 0.95)}}
    examples = {key: [] for key in accumulators}
    seen = 0
    generator = torch.Generator(device="cpu").manual_seed(20260908)

    for batch in loader:
        if seen >= num_samples:
            break
        take = min(len(batch["target"]), num_samples - seen)
        input_ids = torch.from_numpy(np.asarray(batch["input_ids"][:take])).to(device).long()
        encoder_mask = torch.from_numpy(np.asarray(batch["encoder_attention_mask"][:take])).to(device).float()
        attention_mask = torch.from_numpy(np.asarray(batch["attention_mask"][:take])).to(device).float()
        cond_mask = torch.from_numpy(np.asarray(batch["cond_seq_mask"][:take])).to(device).float()
        references = list(batch["target"][:take])
        cond_lengths = cond_mask.to(torch.int32).sum(dim=1)
        target_mask = attention_mask * (1.0 - cond_mask)

        x0 = encode_text(
            input_ids, encoder_mask, encoder, config.latent_mean, config.latent_std,
            use_bf16=bool(config.use_bf16),
        ).to(next(model.parameters()).dtype)
        generation_length = config.max_length - config.max_input_length

        clean_ids = _dlm_decode_batch(x0, model, 1.0, config, 1.0)
        clean_scores, clean_text = score_prediction(
            clean_ids, input_ids, target_mask, cond_lengths, references, tokenizer,
            eos_id, pad_id, generation_length,
        )
        accumulators["oracle_clean"].append((clean_scores, take))
        examples["oracle_clean"].extend(clean_text[:max(0, 8 - len(examples["oracle_clean"]))])

        noise = torch.randn(x0.shape, generator=generator, dtype=x0.dtype).to(device)
        cond_mask_3d = cond_mask.unsqueeze(-1)
        initial_self_cond = torch.where(cond_mask_3d > 0, x0, torch.zeros_like(x0))
        for t_value in (0.25, 0.5, 0.75, 0.95):
            t = torch.full((take,), t_value, device=device, dtype=x0.dtype)
            z = add_noise(x0, noise, t, config, cond_seq_mask=cond_mask_3d)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=device.type == "cuda" and bool(config.use_bf16)):
                _, x_pred = _forward_sample(
                    model=model, z=z, t_batch=t, x_pred_prev=initial_self_cond,
                    config=config, cfg_scale=1.0, self_cond_cfg_scale=1.0,
                    cond_seq=x0, cond_seq_mask=cond_mask,
                )
            pred_ids = _dlm_decode_batch(x_pred, model, 1.0, config, 1.0)
            key = f"denoise_t{t_value}"
            scores, texts = score_prediction(
                pred_ids, input_ids, target_mask, cond_lengths, references, tokenizer,
                eos_id, pad_id, generation_length,
            )
            accumulators[key].append((scores, take))
            examples[key].extend(texts[:max(0, 8 - len(examples[key]))])
        seen += take

    result = {"name": name, "checkpoint": checkpoint_path, "step": step, "samples": seen,
              "metrics": {}, "examples": examples}
    for key, batches in accumulators.items():
        total = sum(weight for _, weight in batches)
        metric_names = batches[0][0].keys()
        result["metrics"][key] = {
            metric: sum(values[metric] * weight for values, weight in batches) / total
            for metric in metric_names if metric != "target_tokens"
        }
        result["metrics"][key]["target_tokens"] = sum(
            values["target_tokens"] for values, _ in batches
        )
    del model
    torch.cuda.empty_cache()
    return result


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config_from_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    dataset = load_dataset_split(config.eval_data_path)
    checkpoints = []
    for spec in args.checkpoint:
        name, path = spec.split("=", 1)
        checkpoints.append((name, path))
    report = {"device": str(device), "diagnostics": []}
    for name, path in checkpoints:
        print(f"Diagnosing {name}: {path}", flush=True)
        result = diagnose_one(
            name, path, config, encoder, encoder_config, tokenizer, dataset, device,
            args.num_samples, args.batch_size,
        )
        report["diagnostics"].append(result)
        print(json.dumps(result["metrics"], indent=2), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
