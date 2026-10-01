#!/usr/bin/env python
"""Position-specific causal transport from semantic table cells into answer velocity."""

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
from official_flow_layer_causal_map import decode_state, prepare_flow
from official_head_position_patching import position_mappings
from official_layerwise_patching import semantic_token_map
from official_reasoning_velocity_transport import advance, cosine, finish, forward_velocity
from official_temporal_causal_tracing import capture_qkv
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split
from utils.sampling_utils import get_sampling_steps


MODES = {
    "k_final_table": ("k", "final_function_table", (1, 2, 3)),
    "k_edited_table_control": ("k", "changed_function_table", (1, 2, 3)),
    "v_changed_cell": ("v", "changed_cell", (4,)),
    "v_swap_partner_control": ("v", "swap_partner", (4,)),
    "v_all_table": ("v", "all_table", (4,)),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True); p.add_argument("--pairs-per-step", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16); p.add_argument("--flow-steps", type=int, default=8)
    p.add_argument("--cfg", type=float, default=2.); p.add_argument("--self-cond-cfg", type=float, default=1.)
    p.add_argument("--block", type=int, default=11); p.add_argument("--head", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260927)
    return p.parse_args()


def semantic_patched_velocity(model, state, replacement_qkv, component, mappings,
                              t, config, cfg, self_cond_cfg, block, head,
                              strength=1.0):
    attention = model.blocks[block - 1].attn; module = attention.qkv
    component_index = {"k": 1, "v": 2}[component]
    prefix = model.num_model_mode_tokens + model.num_time_tokens + model.num_self_cond_cfg_tokens
    calls = 0
    def hook(_module, _inputs, output):
        nonlocal calls
        current = calls; calls += 1
        if current != 0: return output
        bsz, length, _ = output.shape
        patched = output.clone().reshape(bsz, length, 3, attention.num_heads,
                                         attention.dim // attention.num_heads)
        donor = replacement_qkv.reshape_as(patched)
        for row, pairs in enumerate(mappings):
            for source_position, donor_position in pairs:
                source_value = patched[row, source_position + prefix, component_index, head]
                donor_value = donor[
                    row, donor_position + prefix, component_index, head
                ].to(patched.dtype)
                patched[row, source_position + prefix, component_index, head] = (
                    source_value + strength * (donor_value - source_value)
                )
        return patched.reshape_as(output)
    handle = module.register_forward_hook(hook)
    try:
        result = forward_velocity(model, state, t, config, cfg, self_cond_cfg)
    finally:
        handle.remove()
    expected = 1 if cfg == 1. else 2
    if calls != expected: raise RuntimeError(f"Expected {expected} calls, got {calls}")
    return result


def aggregate(rows):
    result = []
    for (mode, time_index), items in sorted(defaultdict(list, {
        key: [r for r in rows if (r["mode"], r["time_index"]) == key]
        for key in set((r["mode"], r["time_index"]) for r in rows)
    }).items()):
        relevant = [r for r in items if r["eligible"] and r["relevant_edit"]]
        result.append({"mode": mode, "time_index": time_index,
            "flow_time": items[0]["flow_time"], "samples": len(items),
            "relevant_samples": len(relevant),
            "velocity_endpoint_alignment": float(np.mean([r["velocity_endpoint_alignment"] for r in relevant])),
            "one_step_donor_progress": float(np.mean([r["one_step_donor_progress"] for r in relevant])),
            "endpoint_projection": float(np.mean([r["endpoint_projection"] for r in relevant])),
            "velocity_shift_norm": float(np.mean([r["velocity_shift_norm"] for r in relevant])),
            "counterfactual_answer_rate": float(np.mean([r["follows_counterfactual"] for r in relevant])),
            "original_answer_rate": float(np.mean([r["follows_original"] for r in relevant])),
        })
    return result


def plot(metrics, output_dir, flow_steps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fields = [("velocity_endpoint_alignment", "Velocity–endpoint alignment"),
              ("one_step_donor_progress", "One-step donor progress"),
              ("endpoint_projection", "Endpoint-direction displacement"),
              ("counterfactual_answer_rate", "Final donor-answer rate")]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5), sharex=True)
    for axis, (field, title) in zip(axes, fields):
        for mode in MODES:
            values = sorted([m for m in metrics if m["mode"] == mode], key=lambda x:x["time_index"])
            axis.plot([m["flow_time"] for m in values], [m[field] for m in values], marker="o", label=mode)
        axis.axhline(0, color="black", linewidth=.8, alpha=.5); axis.grid(alpha=.25)
        axis.set_title(title); axis.set_xlabel("Flow time")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(output_dir / "semantic_velocity_transport.png", dpi=220); plt.close(fig)


@torch.no_grad()
def run(args):
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config_from_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    token_map = semantic_token_map(tokenizer)
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    model, checkpoint_step = load_model(config, args.checkpoint, encoder_config, tokenizer, device)
    validation = load_dataset_split(config.eval_data_path)
    originals, counterfactuals, counts = build_pairs(validation, tokenizer, args.pairs_per_step, args.seed)
    loader_args = dict(batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False,
        max_seq_length=config.max_length, pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
        max_input_seq_length=config.max_input_length, distributed=False)
    source_loader = get_dataloader(originals, **loader_args); donor_loader = get_dataloader(counterfactuals, **loader_args)
    dtype = next(model.parameters()).dtype
    t_steps = get_sampling_steps(args.flow_steps, "uniform", config.denoiser_p_mean,
        config.denoiser_p_std, device=device, dtype=dtype)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    use_bf16 = bool(config.use_bf16) and device.type == "cuda"
    rows = []; cursor = 0
    for source_batch, donor_batch in zip(source_loader, donor_loader):
        bsz = len(source_batch["target"])
        noise = torch.randn((bsz, config.max_length, model.text_encoder_dim), generator=generator, dtype=dtype)
        source = prepare_flow(source_batch, noise, model, encoder, tokenizer, config, device)
        donor = prepare_flow(donor_batch, noise, model, encoder, tokenizer, config, device)
        source_rows = [originals[cursor+i] for i in range(bsz)]
        donor_rows = [counterfactuals[cursor+i] for i in range(bsz)]
        mapping_sets = []
        for i in range(bsz):
            mapping_sets.append(position_mappings(source_rows[i], donor_rows[i],
                source["positions"][i], donor["positions"][i], tokenizer))
        edit_steps = [int(row["intervention_step"]) for row in source_rows]
        source_states=[]; donor_states=[]; source_velocities=[]; donor_qkvs=[]
        for index in range(args.flow_steps):
            source_states.append(source); donor_states.append(donor)
            donor_qkvs.append(capture_qkv(model, donor, t_steps[index], args.block, args.self_cond_cfg, use_bf16))
            sv, sx = forward_velocity(model, source, t_steps[index], config, args.cfg, args.self_cond_cfg)
            dv, dx = forward_velocity(model, donor, t_steps[index], config, args.cfg, args.self_cond_cfg)
            source_velocities.append(sv); dt=t_steps[index+1]-t_steps[index]
            source=advance(source,sv,sx,dt); donor=advance(donor,dv,dx,dt)
        source_final=source; donor_final=donor
        source_stats=decode_state(source_final,model,config,args.self_cond_cfg)
        donor_stats=decode_state(donor_final,model,config,args.self_cond_cfg)
        source_tokens=torch.tensor([token_map[int(x)] for x in source_batch["target"]],device=device)
        donor_tokens=torch.tensor([token_map[int(x)] for x in donor_batch["target"]],device=device)
        positions=source_states[0]["positions"]; batch_rows=torch.arange(bsz,device=device)
        source_endpoint=source_final["z"][batch_rows,positions].float()
        donor_endpoint=donor_final["z"][batch_rows,positions].float()
        endpoint_direction=donor_endpoint-source_endpoint
        for index in range(args.flow_steps):
            dt=t_steps[index+1]-t_steps[index]; state=source_states[index]
            source_v_ans=source_velocities[index][batch_rows,positions].float()
            source_next=(state["z"]+dt*source_velocities[index])[batch_rows,positions].float()
            donor_state=donor_states[index]
            donor_v, donor_x=forward_velocity(model,donor_state,t_steps[index],config,args.cfg,args.self_cond_cfg)
            donor_next=(donor_state["z"]+dt*donor_v)[batch_rows,positions].float()
            for mode,(component,mapping_name,relevant_steps) in MODES.items():
                mappings=[item[mapping_name] for item in mapping_sets]
                pv,px=semantic_patched_velocity(model,state,donor_qkvs[index],component,mappings,
                    t_steps[index],config,args.cfg,args.self_cond_cfg,args.block,args.head)
                patched_next=advance(state,pv,px,dt)
                patched_ans=patched_next["z"][batch_rows,positions].float()
                patched_final=finish(model,patched_next,index+1,t_steps,config,args.cfg,args.self_cond_cfg)
                stats=decode_state(patched_final,model,config,args.self_cond_cfg)
                delta_v=pv[batch_rows,positions].float()-source_v_ans
                endpoint_norm=endpoint_direction.norm(dim=-1).clamp_min(1e-8)
                native_dist=(source_next-donor_next).norm(dim=-1).clamp_min(1e-8)
                progress=(native_dist-(patched_ans-donor_next).norm(dim=-1))/native_dist
                projection=(dt.float()*delta_v*endpoint_direction).sum(dim=-1)/endpoint_norm.square()
                align=cosine(delta_v,endpoint_direction)
                for local in range(bsz):
                    pred=stats["prediction"][local]
                    rows.append({"pair_index":cursor+local,"intervention_step":edit_steps[local],
                        "mode":mode,"component":component,"time_index":index,"flow_time":float(t_steps[index]),
                        "relevant_edit":edit_steps[local] in relevant_steps,
                        "velocity_shift_norm":float(delta_v[local].norm()),
                        "velocity_endpoint_alignment":float(align[local]),
                        "one_step_donor_progress":float(progress[local]),"endpoint_projection":float(projection[local]),
                        "follows_original":bool(pred==source_tokens[local]),
                        "follows_counterfactual":bool(pred==donor_tokens[local]),
                        "eligible":bool(source_stats["correct"][local] and donor_stats["correct"][local])})
        cursor+=bsz; print(f"Semantic velocity transport: processed {cursor}/{len(originals)}",flush=True)
    metrics=aggregate(rows)
    summary={"checkpoint":args.checkpoint,"checkpoint_step":checkpoint_step,"seed":args.seed,
        "counts":counts,"pairs":len(originals),"flow_steps":args.flow_steps,
        "t_steps":[float(x) for x in t_steps],"modes":{k:[v[0],v[1],list(v[2])] for k,v in MODES.items()},
        "metrics":metrics}
    (output_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    with (output_dir/"transport_metrics.csv").open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    with (output_dir/"per_sample.jsonl").open("w") as h:
        for row in rows:h.write(json.dumps(row)+"\n")
    plot(metrics,output_dir,args.flow_steps)
    print(f"Saved semantic velocity transport to {output_dir}",flush=True)


if __name__=="__main__":run(parse_args())
