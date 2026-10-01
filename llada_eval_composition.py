#!/usr/bin/env python3
"""Evaluate LLaDA-Instruct on the held-out function-composition split.

The script deliberately uses the exact validation rows used by the ELF run and
reports accuracy separately by task and reasoning depth.  Generation is
deterministic (temperature zero); every prediction is persisted as JSONL so
that metrics can be independently recomputed.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModel, AutoTokenizer


LLADA_CODE = Path(
    "/vsc-hard-mounts/leuven-data/377/vsc37788/"
    "server-backup-2026-07-29/DiffusionRL-xt/LLaDA"
)
sys.path.insert(0, str(LLADA_CODE))
from generate import generate  # noqa: E402


def first_integer(text: str):
    match = re.search(r"(?<!\d)-?\d+", text)
    return int(match.group()) if match else None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--dataset",
        default=(
            "/vsc-hard-mounts/leuven-data/377/vsc37788/continuous-reasoning/"
            "data/official-composition-v1/validation"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", nargs="+", default=["compose"])
    parser.add_argument("--depths", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--per-group", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--gen-length", type=int, default=8)
    parser.add_argument("--block-length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gen_length % args.block_length:
        raise ValueError("gen-length must be divisible by block-length")
    blocks = args.gen_length // args.block_length
    if args.steps % blocks:
        raise ValueError("steps must be divisible by the number of blocks")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)

    dataset = load_from_disk(args.dataset)
    selected = []
    counts = defaultdict(int)
    wanted_tasks = set(args.tasks)
    wanted_depths = set(args.depths)
    for idx, row in enumerate(dataset):
        key = (row["task"], int(row["depth"]))
        if key[0] not in wanted_tasks or key[1] not in wanted_depths:
            continue
        if counts[key] >= args.per_group:
            continue
        selected.append((idx, row))
        counts[key] += 1

    expected = {(task, depth) for task in args.tasks for depth in args.depths}
    missing = {key: counts[key] for key in expected if counts[key] != args.per_group}
    if missing:
        raise RuntimeError(f"Incomplete evaluation groups: {missing}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id == 126336:
        raise RuntimeError("LLaDA mask token cannot also be the padding token")

    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).cuda().eval()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    correct = defaultdict(int)
    total = defaultdict(int)

    with out_path.open("w", encoding="utf-8") as handle:
        for lo in range(0, len(selected), args.batch_size):
            batch = selected[lo : lo + args.batch_size]
            prompts = []
            for _, row in batch:
                question = (
                    row["input"]
                    + " Return only the single integer answer from 0 through 7, "
                    "with no explanation."
                )
                prompts.append(
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": question}],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                )
            encoded = tokenizer(
                prompts,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].cuda()
            attention_mask = encoded["attention_mask"].cuda()
            output_ids = generate(
                model,
                input_ids,
                attention_mask=attention_mask,
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
                temperature=0.0,
                cfg_scale=0.0,
                remasking="low_confidence",
            )
            decoded = tokenizer.batch_decode(
                output_ids[:, input_ids.shape[1] :], skip_special_tokens=True
            )
            for (dataset_id, row), prediction in zip(batch, decoded):
                key = (row["task"], int(row["depth"]))
                pred_int = first_integer(prediction)
                target_int = int(row["target"])
                is_correct = pred_int == target_int
                total[key] += 1
                correct[key] += int(is_correct)
                record = {
                    "dataset_id": dataset_id,
                    "task": key[0],
                    "depth": key[1],
                    "target": row["target"],
                    "prediction": prediction,
                    "first_integer": pred_int,
                    "correct": is_correct,
                    "steps": args.steps,
                    "gen_length": args.gen_length,
                    "block_length": args.block_length,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            done = min(lo + len(batch), len(selected))
            print(f"evaluated={done}/{len(selected)}", flush=True)

    summary = {}
    for key in sorted(total):
        acc = correct[key] / total[key]
        name = f"{key[0]}_d{key[1]}"
        summary[name] = {"correct": correct[key], "total": total[key], "accuracy": acc}
        print(f"{name}: {correct[key]}/{total[key]} = {100 * acc:.2f}%")
    summary["overall"] = {
        "correct": sum(correct.values()),
        "total": sum(total.values()),
        "accuracy": sum(correct.values()) / sum(total.values()),
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote={out_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
