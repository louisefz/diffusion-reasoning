#!/usr/bin/env python
"""Plot answer decodability jointly across ELF flow time and denoiser depth.

The diagnostic deliberately uses the *same* layer-wise linear probes at every
flow time.  Those probes were fit on the standard t=0 conditional forward, so
the heatmap measures when that fixed answer code becomes readable rather than
fitting a fresh classifier to every heatmap cell.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent
OFFICIAL_SRC = ROOT / "official-elf" / "src"
sys.path.insert(0, str(OFFICIAL_SRC))

from configs.config import load_config_from_yaml
from modules.t5_encoder import get_encoder
from official_diagnostics import load_model
from official_layerwise_probe import HiddenRecorder
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split
from utils.sampling_utils import restore_cond


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--probes", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=30)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config_from_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name or config.encoder_model_name
    )
    encoder_config, _ = get_encoder(config.encoder_model_name, torch.float32)
    model, checkpoint_step = load_model(
        config, args.checkpoint, encoder_config, tokenizer, device,
    )
    dtype = next(model.parameters()).dtype

    payload = torch.load(args.trajectory, map_location="cpu", weights_only=False)
    probes = joblib.load(args.probes)
    if len(probes) != model.depth + 1:
        raise ValueError(f"Expected {model.depth + 1} probes, got {len(probes)}")
    source_ids = payload["source_ids"].long().tolist()
    dataset = load_dataset_split(config.eval_data_path).select(source_ids)
    loader = get_dataloader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        drop_last=False, max_seq_length=config.max_length,
        pad_token_id=get_pad_token_id(tokenizer, config.pad_token),
        max_input_seq_length=config.max_input_length, distributed=False,
    )

    t_steps = payload["t_steps"].float().numpy()
    groups = np.asarray(payload["groups"])
    labels = np.asarray([int(dataset[i]["target"]) for i in range(len(dataset))])
    group_names = ["compose/d1", "compose/d2", "compose/d4", "lookup/d4"]
    correct = np.zeros(
        (len(dataset), len(t_steps), model.depth + 1), dtype=np.bool_
    )
    cursor = 0
    use_bf16 = bool(config.use_bf16) and device.type == "cuda"

    for batch in loader:
        bsz = len(batch["target"])
        sl = slice(cursor, cursor + bsz)
        cond_mask = torch.from_numpy(np.asarray(batch["cond_seq_mask"])).to(device).float()
        answer_positions = payload["answer_positions"][sl].to(device).long()
        z_all = payload["z"][sl]
        for time_index, t_value in enumerate(t_steps):
            z = z_all[:, time_index].to(device=device, dtype=dtype)
            # Standardize self-conditioning across time: the second half only
            # contains the immutable condition, exactly as in probe training.
            xpred_previous = restore_cond(torch.zeros_like(z), z, cond_mask)
            model_input = torch.cat([z, xpred_previous], dim=-1)
            t_batch = torch.full(
                (bsz,), float(t_value), device=device, dtype=dtype,
            )
            sc_batch = torch.ones((bsz,), device=device, dtype=dtype)
            recorder = HiddenRecorder(model, answer_positions)
            try:
                with torch.amp.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=use_bf16,
                ):
                    model(
                        model_input, t_batch, deterministic=True,
                        self_cond_cfg_scale=sc_batch, decoder_step_active=None,
                    )
            finally:
                recorder.close()
            features = torch.stack(recorder.hidden, dim=1).float().cpu().numpy()
            for layer, probe in enumerate(probes):
                prediction = probe.predict(features[:, layer])
                correct[sl, time_index, layer] = prediction == labels[sl]
        cursor += bsz
        print(f"Processed {cursor}/{len(dataset)} trajectories", flush=True)

    maps = {}
    for group in group_names:
        mask = groups == group
        maps[group] = correct[mask].mean(axis=0).T
    maps["all"] = correct.mean(axis=0).T

    np.savez_compressed(
        output_dir / "flow_layer_accuracy.npz",
        t_steps=t_steps,
        layers=np.arange(model.depth + 1),
        **{name.replace("/", "_"): value for name, value in maps.items()},
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(17, 5.1), sharex=True, sharey=True)
    image = None
    for axis, group in zip(axes, group_names):
        values = maps[group]
        image = axis.imshow(
            values, origin="lower", aspect="auto", interpolation="nearest",
            extent=[float(t_steps[0]), float(t_steps[-1]), -0.5, model.depth + 0.5],
            vmin=0.125, vmax=1.0, cmap="magma",
        )
        axis.set_title(group)
        axis.set_xlabel("Flow time $t$")
        axis.set_xticks([0, .25, .5, .75, 1.0])
        axis.set_yticks(range(model.depth + 1))
        axis.axhline(8.5, color="cyan", linewidth=.8, alpha=.65)
    axes[0].set_ylabel("Representation depth (0=input, 1–12=blocks)")
    colorbar = fig.colorbar(image, ax=axes, fraction=.022, pad=.025)
    colorbar.set_label("Fixed-probe held-out answer accuracy")
    fig.suptitle(
        "Where answer information is readable across ELF flow time and depth",
        y=.995, fontsize=13,
    )
    fig.subplots_adjust(left=.06, right=.91, bottom=.12, top=.88, wspace=.08)
    png = output_dir / "flow_time_x_transformer_depth.png"
    pdf = output_dir / "flow_time_x_transformer_depth.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": checkpoint_step,
        "trajectory": args.trajectory,
        "probes": args.probes,
        "samples": len(dataset),
        "samples_per_group": {g: int((groups == g).sum()) for g in group_names},
        "metric": "accuracy of fixed layer-wise Ridge probes trained at t=0",
        "diagnostic_forward": "conditional model at each saved z_t; self-conditioning reset to condition-only",
        "t_steps": t_steps.tolist(),
        "png": str(png),
        "pdf": str(pdf),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
