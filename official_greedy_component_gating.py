#!/usr/bin/env python
"""Greedily construct a jointly safe late-flow component-gating policy."""

import argparse
import json
import sys
from pathlib import Path

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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True); p.add_argument("--samples-per-group", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=20); p.add_argument("--flow-steps", type=int, default=8)
    p.add_argument("--cutoff", type=float, default=.625); p.add_argument("--cfg", type=float, default=2.)
    p.add_argument("--self-cond-cfg", type=float, default=1.); p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--fixed-policy", default=None,
                   help="Comma-separated block:component entries; validate only, without search")
    return p.parse_args()


def gated_step(model, state, t, t_next, config, cfg, self_cond_cfg, policy):
    calls = {item: 0 for item in policy}; handles = []
    for block_number, component in policy:
        module = getattr(model.blocks[block_number - 1], component)
        def hook(_module, _inputs, output, item=(block_number, component)):
            calls[item] += 1
            return torch.zeros_like(output)
        handles.append(module.register_forward_hook(hook))
    try:
        result = ode_step(model, state, t, t_next, config, cfg, self_cond_cfg)
    finally:
        for handle in handles: handle.remove()
    expected = 1 if cfg == 1. else 2
    if any(count != expected for count in calls.values()):
        raise RuntimeError(f"Unexpected hook calls: {calls}")
    return result


@torch.no_grad()
def run(args):
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    if abs(float(t_steps[cutoff_index]) - args.cutoff) > 1e-5: raise ValueError("cutoff not on schedule")
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    records = []; cursor = 0
    for batch in loader:
        bsz = len(batch["target"])
        noise = torch.randn((bsz, config.max_length, model.text_encoder_dim), generator=generator, dtype=dtype)
        state = prepare_flow(batch, noise, model, encoder, tokenizer, config, device)
        cutoff_state = None
        for step in range(args.flow_steps):
            if step == cutoff_index: cutoff_state = state
            state = ode_step(model, state, t_steps[step], t_steps[step + 1], config, args.cfg, args.self_cond_cfg)
        native = decode_state(state, model, config, args.self_cond_cfg)
        records.append((cutoff_state, list(native["prediction"]), groups[cursor:cursor + bsz]))
        cursor += bsz

    def evaluate(policy):
        total = agree = 0; by_group = {}
        for cutoff_state, native_prediction, batch_groups in records:
            state = dict(cutoff_state)
            for step in range(cutoff_index, args.flow_steps):
                state = gated_step(model, state, t_steps[step], t_steps[step + 1], config,
                                   args.cfg, args.self_cond_cfg, policy)
            pred = decode_state(state, model, config, args.self_cond_cfg)["prediction"]
            for p, n, group in zip(pred, native_prediction, batch_groups):
                same = bool(p == n)
                total += 1; agree += int(same)
                stats = by_group.setdefault(group, [0, 0]); stats[1] += 1; stats[0] += int(same)
        return agree / total, {g: a / n for g, (a, n) in by_group.items()}

    if args.fixed_policy:
        policy = []
        for item in args.fixed_policy.split(","):
            block, component = item.split(":")
            policy.append((int(block), component))
        agreement, by_group = evaluate(policy)
        summary = {"checkpoint": args.checkpoint, "checkpoint_step": checkpoint_step,
            "cutoff": args.cutoff, "flow_steps": args.flow_steps, "seed": args.seed,
            "samples_per_group": args.samples_per_group, "source_ids": source_ids,
            "fixed_policy": [list(x) for x in policy],
            "native_answer_agreement": agreement, "agreement_by_group": by_group,
            "warning": "hooks validate functional equivalence but do not save real compute"}
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("FIXED POLICY", policy, agreement, by_group, flush=True)
        return

    mlps = [(b, "mlp") for b in (2, 3, 4, 5, 6, 7, 8, 9, 10, 1)]
    attns = [(b, "attn") for b in (6, 5, 4, 7, 8, 9, 2, 1, 10, 11)]
    orders = {"mlp_first": mlps + attns, "attn_first": attns + mlps,
              "interleaved": [x for pair in zip(mlps, attns) for x in pair]}
    searches = {}
    for name, candidates in orders.items():
        accepted = []; trace = []
        for candidate in candidates:
            trial = accepted + [candidate]
            agreement, by_group = evaluate(trial)
            keep = agreement == 1.0
            if keep: accepted = trial
            trace.append({"candidate": list(candidate), "kept": keep,
                          "native_answer_agreement": agreement, "agreement_by_group": by_group,
                          "accepted_after": [list(x) for x in accepted]})
            print(name, candidate, "keep" if keep else "reject", agreement, flush=True)
        final_agreement, final_by_group = evaluate(accepted)
        searches[name] = {"accepted": [list(x) for x in accepted], "count": len(accepted),
                          "native_answer_agreement": final_agreement,
                          "agreement_by_group": final_by_group, "trace": trace}
    best_name = max(searches, key=lambda k: searches[k]["count"])
    summary = {"checkpoint": args.checkpoint, "checkpoint_step": checkpoint_step,
        "cutoff": args.cutoff, "flow_steps": args.flow_steps,
        "samples_per_group": args.samples_per_group, "source_ids": source_ids,
        "criterion": "strict 100% native answer-token agreement on all search samples",
        "warning": "search-set result requires held-out validation; hooks do not save real compute",
        "searches": searches, "best_order": best_name, "best_policy": searches[best_name]["accepted"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("BEST", best_name, searches[best_name]["accepted"], flush=True)


if __name__ == "__main__": run(parse_args())
