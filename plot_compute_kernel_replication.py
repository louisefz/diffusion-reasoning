#!/usr/bin/env python
"""Combine recurrent-compute impulse response kernels across schedules/seeds."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUNS = [
    (62167191, "logit-normal seed 0"),
    (62167203, "logit-normal seed 1"),
    (62167204, "logit-normal seed 2"),
    (62167202, "uniform schedule"),
]


def load(job):
    path = Path(f"runs/official-recurrent-compute-impulse-{job}/summary.json")
    report = json.loads(path.read_text())
    rows = {row["name"]: row for row in report["results"]}
    baseline = rows["all_K1"]["accuracy"]
    times = np.asarray(report["outer_times"])
    effects = 100 * np.asarray([
        rows[f"impulse_{index}"]["accuracy"] - baseline
        for index in range(len(times))
    ])
    return times, effects


def main():
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), constrained_layout=True)
    early, late, labels = [], [], []
    for job, label in RUNS:
        times, effects = load(job)
        axes[0].plot(times, effects, "o-", lw=2, alpha=0.9, label=label)
        early.append(effects[:4].mean())
        late.append(effects[-4:].mean())
        labels.append(label.replace("logit-normal ", "LN "))
    axes[0].axhline(0, color="black", ls=":", lw=1.3)
    axes[0].set(xlabel="Flow time of one K1→K4 impulse",
                ylabel="Endpoint accuracy change (percentage points)",
                title="Compute-response kernels replicate")
    axes[0].legend(frameon=False, fontsize=8)

    x = np.arange(len(labels))
    width = 0.36
    axes[1].bar(x - width / 2, early, width, label="mean earliest 4 calls")
    axes[1].bar(x + width / 2, late, width, label="mean latest 4 calls")
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].set_ylabel("Mean endpoint gain (percentage points)")
    axes[1].set_title("Early compute has greater leverage")
    axes[1].legend(frameon=False, fontsize=8)
    output = Path("analysis_artifacts/recurrent_compute_response_kernel_replication.png")
    fig.savefig(output, dpi=220)
    print(output)


if __name__ == "__main__":
    main()
