#!/usr/bin/env python
"""Render existing flow×layer diagnostic matrices with honest time geometry."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GROUPS = ["compose/d1", "compose/d2", "compose/d4", "lookup/d4"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def finish(fig, axes, image, output):
    axes[0].set_ylabel("Representation depth (0=input, 1–12=blocks)")
    colorbar = fig.colorbar(image, ax=axes, fraction=.022, pad=.025)
    colorbar.set_label("Fixed-probe held-out answer accuracy")
    fig.suptitle(
        "Where answer information is readable across ELF flow time and depth",
        y=.995, fontsize=13,
    )
    fig.subplots_adjust(left=.06, right=.91, bottom=.14, top=.88, wspace=.08)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    data = np.load(args.input)
    t_steps = data["t_steps"]
    layers = data["layers"]
    maps = {group: data[group.replace("/", "_")] for group in GROUPS}

    # Literal flow-time geometry: columns are placed at their true t values.
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.1), sharex=True, sharey=True)
    for axis, group in zip(axes, GROUPS):
        image = axis.pcolormesh(
            t_steps, layers, maps[group], shading="nearest",
            vmin=.125, vmax=1., cmap="magma",
        )
        axis.set_title(group)
        axis.set_xlabel("Flow time $t$ (true spacing)")
        axis.set_xlim(0, 1)
        axis.set_xticks([0, .25, .5, .75, 1.])
        axis.set_yticks(layers)
        axis.axhline(8.5, color="cyan", linewidth=.8, alpha=.65)
    finish(fig, axes, image, output_dir / "flow_time_x_transformer_depth.png")

    # Readable schedule view: equal-width columns, with actual t values shown.
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.1), sharex=True, sharey=True)
    tick_index = np.asarray([0, 4, 8, 12, 16])
    tick_label = [f"{t_steps[index]:.2f}" for index in tick_index]
    for axis, group in zip(axes, GROUPS):
        image = axis.imshow(
            maps[group], origin="lower", aspect="auto", interpolation="nearest",
            vmin=.125, vmax=1., cmap="magma",
        )
        axis.set_title(group)
        axis.set_xlabel("Flow step (tick label = $t$)")
        axis.set_xticks(tick_index, tick_label)
        axis.set_yticks(layers)
        axis.axhline(8.5, color="cyan", linewidth=.8, alpha=.65)
    finish(fig, axes, image, output_dir / "flow_step_x_transformer_depth.png")


if __name__ == "__main__":
    main()
