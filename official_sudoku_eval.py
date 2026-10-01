#!/usr/bin/env python
"""Verifier-backed held-out evaluation for the official ELF 9x9 Sudoku model."""

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
from modules.t5_encoder import get_encoder
from official_diagnostics import load_model
from sudoku9_task import verify
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split
from utils.encoder_utils import encode_text
from utils.generation_utils import _dlm_decode_batch, _generate_samples_single_batch
from utils.sampling_utils import get_sampling_steps


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--num-samples", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--cfgs", nargs="+", type=float, default=[1, 2])
    parser.add_argument("--seed", type=int, default=20260928)
    return parser.parse_args()


def parse_grid(text):
    try:
        values = [int(token) for token in text.strip().split()]
    except ValueError:
        return None
    if len(values) != 81 or any(value < 1 or value > 9 for value in values):
        return None
    return values


@torch.no_grad()
def evaluate(model, encoder, dataset, tokenizer, config, device, steps, cfg,
             num_samples, batch_size, seed):
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
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    dtype = next(model.parameters()).dtype
    seen = token_correct = token_total = 0
    exact = valid = parsed = cell_correct = cell_total = 0
    clue_correct = clue_total = blank_correct = blank_total = 0
    grouped = defaultdict(lambda: {
        "samples": 0, "token_correct": 0, "token_total": 0,
        "exact": 0, "valid": 0, "parsed": 0,
        "cell_correct": 0, "cell_total": 0,
        "clue_correct": 0, "clue_total": 0,
        "blank_correct": 0, "blank_total": 0,
    })
    examples = []

    for batch in loader:
        if seen >= num_samples:
            break
        take = min(len(batch["target"]), num_samples - seen)
        input_ids = torch.from_numpy(np.asarray(batch["input_ids"][:take])).to(device).long()
        encoder_mask = torch.from_numpy(np.asarray(batch["encoder_attention_mask"][:take])).to(device).float()
        attention_mask = torch.from_numpy(np.asarray(batch["attention_mask"][:take])).to(device).float()
        cond_mask = torch.from_numpy(np.asarray(batch["cond_seq_mask"][:take])).to(device).float()
        starts = cond_mask.long().sum(dim=1)
        target_mask = (attention_mask * (1.0 - cond_mask)).bool()
        lengths = target_mask.long().sum(dim=1)
        cond_seq = encode_text(
            input_ids, encoder_mask, encoder, config.latent_mean,
            config.latent_std, use_bf16=bool(config.use_bf16),
        ).to(dtype)
        z = (torch.randn(
            (take, config.max_length, model.text_encoder_dim),
            generator=generator, dtype=dtype,
        ) * config.denoiser_noise_scale).to(device)
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
        matches = predicted.eq(input_ids)
        metadata = dataset[seen:seen + take]
        for row in range(take):
            start, length = int(starts[row]), int(lengths[row])
            row_token_correct = int(matches[row][target_mask[row]].sum())
            row_token_total = int(target_mask[row].sum())
            is_exact = row_token_correct == row_token_total
            text = tokenizer.decode(
                predicted[row, start:start + length].detach().cpu().tolist(),
                skip_special_tokens=True,
            ).strip()
            grid = parse_grid(text)
            answer = list(map(int, metadata["answer"][row]))
            puzzle = list(map(int, metadata["puzzle"][row]))
            is_parsed = grid is not None
            is_valid = bool(grid is not None and verify({"puzzle": puzzle, "answer": answer}, grid))
            row_cell_correct = sum(a == b for a, b in zip(grid or [], answer))
            row_cell_total = 81 if grid is not None else 0
            if grid is not None:
                row_clue_correct = sum(
                    prediction == target
                    for prediction, target, given in zip(grid, answer, puzzle)
                    if given != 0
                )
                row_clue_total = sum(given != 0 for given in puzzle)
                row_blank_correct = sum(
                    prediction == target
                    for prediction, target, given in zip(grid, answer, puzzle)
                    if given == 0
                )
                row_blank_total = sum(given == 0 for given in puzzle)
            else:
                row_clue_correct = row_clue_total = 0
                row_blank_correct = row_blank_total = 0
            difficulty = str(metadata["difficulty"][row])
            group = grouped[difficulty]
            group["samples"] += 1
            group["token_correct"] += row_token_correct
            group["token_total"] += row_token_total
            group["exact"] += int(is_exact)
            group["valid"] += int(is_valid)
            group["parsed"] += int(is_parsed)
            group["cell_correct"] += row_cell_correct
            group["cell_total"] += row_cell_total
            group["clue_correct"] += row_clue_correct
            group["clue_total"] += row_clue_total
            group["blank_correct"] += row_blank_correct
            group["blank_total"] += row_blank_total
            token_correct += row_token_correct; token_total += row_token_total
            exact += int(is_exact); valid += int(is_valid); parsed += int(is_parsed)
            cell_correct += row_cell_correct; cell_total += row_cell_total
            clue_correct += row_clue_correct; clue_total += row_clue_total
            blank_correct += row_blank_correct; blank_total += row_blank_total
            if len(examples) < 12:
                examples.append({
                    "difficulty": difficulty,
                    "prediction": text,
                    "parsed": is_parsed,
                    "valid": is_valid,
                    "exact": is_exact,
                    "cell_accuracy": row_cell_correct / max(row_cell_total, 1),
                    "clue_accuracy": row_clue_correct / max(row_clue_total, 1),
                    "blank_accuracy": row_blank_correct / max(row_blank_total, 1),
                })
        seen += take

    def finalize(values):
        return {
            "samples": values["samples"],
            "target_token_accuracy": values["token_correct"] / max(values["token_total"], 1),
            "parse_rate": values["parsed"] / max(values["samples"], 1),
            "cell_accuracy_given_parse": values["cell_correct"] / max(values["cell_total"], 1),
            "clue_accuracy": values["clue_correct"] / max(values["clue_total"], 1),
            "blank_accuracy": values["blank_correct"] / max(values["blank_total"], 1),
            "exact_grid_accuracy": values["exact"] / max(values["samples"], 1),
            "verifier_accuracy": values["valid"] / max(values["samples"], 1),
        }

    return {
        "steps": steps, "cfg": cfg, "samples": seen,
        "target_token_accuracy": token_correct / max(token_total, 1),
        "parse_rate": parsed / max(seen, 1),
        "cell_accuracy_given_parse": cell_correct / max(cell_total, 1),
        "clue_accuracy": clue_correct / max(clue_total, 1),
        "blank_accuracy": blank_correct / max(blank_total, 1),
        "exact_grid_accuracy": exact / max(seen, 1),
        "verifier_accuracy": valid / max(seen, 1),
        "groups": {name: finalize(values) for name, values in sorted(grouped.items())},
        "examples": examples,
    }


def main():
    args = parse_args(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config_from_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    model, checkpoint_step = load_model(config, args.checkpoint, encoder_config, tokenizer, device)
    dataset = load_dataset_split(args.data_path or config.eval_data_path)
    report = {"checkpoint_step": checkpoint_step, "results": []}
    for steps, cfg in itertools.product(args.steps, args.cfgs):
        print(f"Evaluating steps={steps}, cfg={cfg}", flush=True)
        result = evaluate(model, encoder, dataset, tokenizer, config, device,
                          steps, cfg, args.num_samples, args.batch_size, args.seed)
        report["results"].append(result); print(json.dumps(result), flush=True)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
