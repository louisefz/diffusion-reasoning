#!/usr/bin/env python
"""Create cross-depth summary figures for the causal circuit response runs."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RUN = 62165962
KERNEL = {
    depth: ROOT / f"runs/official-circuit-depth-d{depth}-kernel-full-{RUN}/summary.json"
    for depth in (1, 2, 4)
}
DOSE = {
    depth: ROOT / f"runs/official-circuit-depth-d{depth}-dose-full-{RUN}/summary.json"
    for depth in (1, 2, 4)
}
SUSTAINED = {
    depth: ROOT / f"runs/official-sustained-force-d{depth}-full-62166039/summary.json"
    for depth in (1, 2, 4)
}
FIXED_RELEASE = {
    depth: ROOT / f"runs/official-fixed-release-d{depth}-full-62166063/summary.json"
    for depth in (1, 2, 4)
}
OUT = ROOT / "analysis_artifacts"
OUT.mkdir(exist_ok=True)


def select(summary, intervention_index, strength=1.0):
    rows = [row for row in summary["metrics"]
            if row["mode"] == "causal_circuit"
            and row["strength"] == strength
            and row["intervention_index"] == intervention_index]
    return sorted(rows, key=lambda row: row["observation_index"])


kernel = {depth: json.loads(path.read_text()) for depth, path in KERNEL.items()}
dose = {depth: json.loads(path.read_text()) for depth, path in DOSE.items()}
sustained = {depth: json.loads(path.read_text()) for depth, path in SUSTAINED.items()}
fixed_release = {depth: json.loads(path.read_text()) for depth, path in FIXED_RELEASE.items()}
colors = {1: "#4c78a8", 2: "#f58518", 4: "#e45756"}

fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
for depth in (1, 2, 4):
    summary = kernel[depth]
    # Exclude the final pulse at s=.875: it has no subsequent native flow step,
    # so final/immediate retention is trivially one and is not a persistence test.
    times = [summary["t_steps"][index] for index in range(7)]
    curves = [select(summary, index) for index in range(7)]
    immediate = np.asarray([rows[0]["answer_axis_projection"] for rows in curves])
    final = np.asarray([rows[-1]["answer_axis_projection"] for rows in curves])
    retention = np.divide(final, immediate, out=np.zeros_like(final),
                          where=np.abs(immediate) > .02)
    donor = np.asarray([rows[-1]["donor_answer_rate"] for rows in curves])
    for axis, values, title in (
        (axes[0], immediate, "Immediate donor-axis response"),
        (axes[1], final, "Endpoint donor-axis response"),
        (axes[2], retention, "Semantic retention (final / immediate)"),
        (axes[3], donor, "Final donor-answer rate"),
    ):
        axis.plot(times, values, marker="o", linewidth=2.2,
                  color=colors[depth], label=f"d{depth}")
        axis.set_title(title); axis.set_xlabel("Intervention flow time s")
        axis.grid(alpha=.25)
axes[2].axhline(0, color="black", linewidth=.8)
axes[3].set_ylim(-.02, 1.02)
axes[0].set_ylabel("Response")
axes[0].legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "circuit_depth_response_comparison.png", dpi=240)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
for depth in (1, 2, 4):
    summary = dose[depth]
    strengths = summary["strengths"]
    curves = [select(summary, 0, strength) for strength in strengths]
    immediate = [rows[0]["answer_axis_projection"] for rows in curves]
    final = [rows[-1]["answer_axis_projection"] for rows in curves]
    donor = [rows[-1]["donor_answer_rate"] for rows in curves]
    for axis, values, title in (
        (axes[0], immediate, "Immediate response at s=0"),
        (axes[1], final, "Endpoint response at s=0"),
        (axes[2], donor, "Donor-answer rate at s=0"),
    ):
        axis.plot(strengths, values, marker="o", linewidth=2.2,
                  color=colors[depth], label=f"d{depth}")
        axis.set_title(title); axis.set_xlabel("Circuit patch strength")
        axis.grid(alpha=.25)
axes[2].set_ylim(-.02, 1.02)
axes[0].legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "circuit_depth_dose_comparison.png", dpi=240)
plt.close(fig)

fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
for depth in (1, 2, 4):
    variable = [row for row in sustained[depth]["final_summary"]
                if row["mode"] == "causal_circuit"]
    fixed = [row for row in fixed_release[depth]["final_summary"]
             if row["mode"] == "causal_circuit"]
    durations = [row["duration"] for row in fixed]
    for axis, values, title in (
        (axes[0], [row["final_donor_answer_rate"] for row in variable],
         "Variable release: donor-answer rate"),
        (axes[1], [row["release_projection"] for row in fixed],
         "Fixed release: state at release"),
        (axes[2], [row["final_projection"] for row in fixed],
         "Fixed release: endpoint response"),
        (axes[3], [row["final_donor_answer_rate"] for row in fixed],
         "Fixed release: donor-answer rate"),
    ):
        axis.plot(durations, values, marker="o", linewidth=2.2,
                  color=colors[depth], label=f"d{depth}")
        axis.set_title(title); axis.set_xlabel("Number of forced steps")
        axis.grid(alpha=.25)
axes[0].set_ylabel("Response")
axes[0].legend(frameon=False)
axes[0].set_ylim(-.02, 1.02); axes[3].set_ylim(-.02, 1.02)
fig.tight_layout()
fig.savefig(OUT / "sustained_forcing_depth_comparison.png", dpi=240)
plt.close(fig)

summary_rows = []
for depth in (1, 2, 4):
    rows = select(kernel[depth], 0)
    immediate = rows[0]["answer_axis_projection"]
    final = rows[-1]["answer_axis_projection"]
    summary_rows.append({
        "depth": depth, "block": kernel[depth]["block"],
        "heads": kernel[depth]["circuit_heads"],
        "immediate_response": immediate, "final_response": final,
        "semantic_retention": final / immediate,
        "response_gain": rows[-1]["response_gain"],
        "donor_answer_rate": rows[-1]["donor_answer_rate"],
    })
(OUT / "circuit_depth_summary.json").write_text(
    json.dumps(summary_rows, indent=2) + "\n")
print(json.dumps(summary_rows, indent=2))
