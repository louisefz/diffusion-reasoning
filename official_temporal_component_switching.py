#!/usr/bin/env python
"""Bidirectional temporal control of ELF head-3 K/V components.

Three protocols expose temporal commitment and repair:
  source_to_donor: native source computation before a switch, donor component after;
  donor_to_native: donor component before a switch, native computation after;
  donor_to_source: donor component before a switch, clean source component after.
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
from official_flow_layer_causal_map import decode_state, ode_step, prepare_flow
from official_layerwise_patching import semantic_token_map
from official_temporal_causal_tracing import capture_qkv, patched_component_step
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split
from utils.sampling_utils import get_sampling_steps


PROTOCOLS = ("source_to_donor", "donor_to_native", "donor_to_source")


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
    parser.add_argument("--seed", type=int, default=20260921)
    return parser.parse_args()


def aggregate(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["component"], row["intervention_step"],
                 row["protocol"], row["switch_index"])].append(row)
    result = []
    for (component, edit_step, protocol, switch_index), items in sorted(buckets.items()):
        eligible = [item for item in items if item["eligible"]]
        selected = eligible or items
        result.append({
            "component": component, "intervention_step": edit_step,
            "protocol": protocol, "switch_index": switch_index,
            "switch_time": items[0]["switch_time"],
            "samples": len(items), "eligible_samples": len(eligible),
            "counterfactual_answer_rate": float(np.mean([
                item["follows_counterfactual"] for item in selected
            ])),
            "original_answer_rate": float(np.mean([
                item["follows_original"] for item in selected
            ])),
            "other_answer_rate": float(np.mean([
                item["follows_other"] for item in selected
            ])),
        })
    return result


def plot(rows, output_dir, flow_steps, t_steps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subsets = {
        "k": [row for row in rows if row["component"] == "k"
              and row["intervention_step"] in (1, 2, 3) and row["eligible"]],
        "v": [row for row in rows if row["component"] == "v"
              and row["intervention_step"] == 4 and row["eligible"]],
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    x = np.arange(flow_steps + 1)
    labels = [f"{float(t_steps[i]):.2f}" for i in range(flow_steps + 1)]
    for axis, component in zip(axes, ("k", "v")):
        selected = subsets[component]
        for protocol in PROTOCOLS:
            values = []
            for switch in range(flow_steps + 1):
                chosen = [row for row in selected if row["protocol"] == protocol
                          and row["switch_index"] == switch]
                values.append(np.mean([row["follows_counterfactual"] for row in chosen]))
            axis.plot(x, values, marker="o", label=protocol.replace("_", " → "))
        label = "edits s1–s3" if component == "k" else "edit s4"
        axis.set_title(f"Block-11 head-3 {component.upper()} ({label})")
        axis.set_xlabel("Switch time")
        axis.set_xticks(x, labels, rotation=30)
        axis.set_ylim(-.03, 1.03); axis.grid(alpha=.25)
        axis.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("Final counterfactual-answer rate")
    fig.tight_layout()
    fig.savefig(output_dir / "temporal_component_switching.png", dpi=220)
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
        source = prepare_flow(source_batch, noise, model, encoder, tokenizer, config, device)
        donor = prepare_flow(donor_batch, noise, model, encoder, tokenizer, config, device)
        source_initial = source
        source_qkv, donor_qkv = [], []
        for step_index in range(args.flow_steps):
            source_qkv.append(capture_qkv(
                model, source, t_steps[step_index], args.block,
                args.self_cond_cfg, use_bf16,
            ))
            donor_qkv.append(capture_qkv(
                model, donor, t_steps[step_index], args.block,
                args.self_cond_cfg, use_bf16,
            ))
            source = ode_step(
                model, source, t_steps[step_index], t_steps[step_index + 1],
                config, args.cfg, args.self_cond_cfg,
            )
            donor = ode_step(
                model, donor, t_steps[step_index], t_steps[step_index + 1],
                config, args.cfg, args.self_cond_cfg,
            )
        source_baseline = decode_state(source, model, config, args.self_cond_cfg)
        donor_baseline = decode_state(donor, model, config, args.self_cond_cfg)
        source_tokens = torch.tensor(
            [token_map[int(x)] for x in source_batch["target"]], device=device,
        )
        donor_tokens = torch.tensor(
            [token_map[int(x)] for x in donor_batch["target"]], device=device,
        )
        edit_steps = list(map(int, originals[cursor:cursor + bsz]["intervention_step"]))

        for component in ("k", "v"):
            for protocol in PROTOCOLS:
                for switch_index in range(args.flow_steps + 1):
                    state = dict(source_initial)
                    for step_index in range(args.flow_steps):
                        replacement = None
                        if protocol == "source_to_donor":
                            if step_index >= switch_index:
                                replacement = donor_qkv[step_index]
                        elif protocol == "donor_to_native":
                            if step_index < switch_index:
                                replacement = donor_qkv[step_index]
                        elif protocol == "donor_to_source":
                            replacement = (
                                donor_qkv[step_index] if step_index < switch_index
                                else source_qkv[step_index]
                            )
                        if replacement is None:
                            state = ode_step(
                                model, state, t_steps[step_index], t_steps[step_index + 1],
                                config, args.cfg, args.self_cond_cfg,
                            )
                        else:
                            state = patched_component_step(
                                model, state, replacement, component,
                                t_steps[step_index], t_steps[step_index + 1],
                                config, args.cfg, args.self_cond_cfg,
                                args.block, args.head,
                            )
                    stats = decode_state(state, model, config, args.self_cond_cfg)
                    for local in range(bsz):
                        prediction = stats["prediction"][local]
                        follows_source = bool(prediction == source_tokens[local])
                        follows_donor = bool(prediction == donor_tokens[local])
                        rows.append({
                            "pair_index": cursor + local,
                            "intervention_step": edit_steps[local],
                            "component": component, "protocol": protocol,
                            "switch_index": switch_index,
                            "switch_time": float(t_steps[switch_index]),
                            "follows_original": follows_source,
                            "follows_counterfactual": follows_donor,
                            "follows_other": not follows_source and not follows_donor,
                            "source_baseline_correct": bool(source_baseline["correct"][local]),
                            "donor_baseline_correct": bool(donor_baseline["correct"][local]),
                            "eligible": bool(
                                source_baseline["correct"][local]
                                and donor_baseline["correct"][local]
                            ),
                        })
        cursor += bsz
        print(f"Temporal switching: processed {cursor}/{len(originals)} pairs", flush=True)

    metrics = aggregate(rows)
    summary = {
        "checkpoint": args.checkpoint, "checkpoint_step": checkpoint_step,
        "device": str(device), "seed": args.seed, "counts": counts,
        "flow_steps": args.flow_steps, "schedule": "uniform",
        "t_steps": [float(x) for x in t_steps], "block": args.block,
        "head": args.head, "protocols": list(PROTOCOLS),
        "source_to_donor": "native source before switch; paired donor component at every step from switch onward",
        "donor_to_native": "paired donor component before switch; native source-conditioned computation afterward",
        "donor_to_source": "paired donor component before switch; clean source-baseline component at every step afterward",
        "relevant_subsets": {"k": "edits s1-s3", "v": "edit s4"},
        "source_baseline_accuracy": float(np.mean([r["source_baseline_correct"] for r in rows])),
        "donor_baseline_accuracy": float(np.mean([r["donor_baseline_correct"] for r in rows])),
        "aggregate": metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "switching_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader(); writer.writerows(metrics)
    with (output_dir / "per_sample.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    plot(rows, output_dir, args.flow_steps, t_steps)
    print(json.dumps({
        "source_baseline_accuracy": summary["source_baseline_accuracy"],
        "donor_baseline_accuracy": summary["donor_baseline_accuracy"],
        "rows": len(rows),
    }, indent=2), flush=True)
    print(f"Saved temporal switching results to {output_dir}", flush=True)


if __name__ == "__main__":
    run(parse_args())
