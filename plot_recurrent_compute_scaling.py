#!/usr/bin/env python
"""Plot held-out accuracy against recurrent reasoning compute."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = json.loads(Path(args.input).read_text())
    rows = sorted(report["results"], key=lambda row: row["reasoning_loops"])
    loops = [row["reasoning_loops"] for row in rows]

    def accuracy(group=None):
        if group is None:
            return [100 * row["first_target_token_accuracy"] for row in rows]
        return [
            100 * row["groups"][group]["first_target_token_accuracy"]
            for row in rows
        ]

    lookup = []
    for row in rows:
        values = [
            value["first_target_token_accuracy"]
            for name, value in row["groups"].items()
            if name.startswith("lookup/")
        ]
        lookup.append(100 * sum(values) / len(values))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)

    ax = axes[0]
    ax.plot(loops, accuracy(), "o-", lw=2.2, label="Overall")
    ax.plot(loops, accuracy("compose/d2"), "o-", lw=2.2, label="Composition d=2")
    ax.plot(loops, accuracy("compose/d4"), "o-", lw=2.8, label="Composition d=4")
    ax.plot(loops, lookup, "o--", lw=1.8, label="Matched lookup")
    ax.axvspan(4, 6, alpha=0.10, color="tab:green", label="best compute region")
    ax.set(xlabel="Inner recurrent steps K", ylabel="Held-out exact accuracy (%)",
           title="Useful compute accumulates, then saturates")
    ax.set_xticks(loops)
    ax.set_ylim(0, 104)
    ax.legend(frameon=True, fontsize=8)

    ax = axes[1]
    for group, label in [
        ("compose/d4", "trained depth d=4"),
        ("compose/d8", "unseen d=8"),
        ("compose/d12", "unseen d=12"),
        ("compose/d16", "unseen d=16"),
    ]:
        ax.plot(loops, accuracy(group), "o-", lw=2.2, label=label)
    ax.axhline(12.5, color="black", ls=":", lw=1.5, label="8-way chance")
    ax.axvline(4, color="tab:green", ls="--", lw=1.5)
    ax.annotate("peak / saturation", xy=(4, 51.5), xytext=(7.2, 65),
                arrowprops={"arrowstyle": "->", "color": "tab:green"},
                color="tab:green", fontsize=9)
    ax.set(xlabel="Inner recurrent steps K", ylabel="Held-out exact accuracy (%)",
           title="Scaling is learned, not depth-general")
    ax.set_xticks(loops)
    ax.set_ylim(0, 75)
    ax.legend(frameon=True, fontsize=8)

    fig.suptitle("Causal-memory recurrent reasoning at fixed 16-step outer flow",
                 fontsize=13, fontweight="bold")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    print(output)


if __name__ == "__main__":
    main()
