#!/usr/bin/env python
"""Validate a flow-recording run and distill its main descriptive findings."""

import argparse
import json
from pathlib import Path

import torch


EXPECTED_GROUPS = ("compose/d1", "compose/d2", "compose/d4", "lookup/d4")
EXPECTED_SAMPLES = 30
EXPECTED_STATES = 17
EXPECTED_STEPS = 16


def first_crossing(rows, key, threshold):
    for row in rows:
        value = row.get(key)
        if value is not None and value >= threshold:
            return {"t": row["t_state"], "index": rows.index(row), "value": value}
    return None


def trapezoid_auc(rows, key):
    usable = [(row["t_state"], row.get(key)) for row in rows]
    usable = [(t, value) for t, value in usable if value is not None]
    return sum(
        (right_t - left_t) * (left_v + right_v) / 2
        for (left_t, left_v), (right_t, right_v) in zip(usable, usable[1:])
    )


def validate(run_dir):
    run_dir = Path(run_dir)
    required = (
        "summary.json", "per_sample.jsonl", "trajectory.csv",
        "accuracy_vs_flow.png", "velocity_vs_flow.png", "trajectory_tensors.pt",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing artifacts: {missing}")
    empty = [name for name in required if (run_dir / name).stat().st_size == 0]
    if empty:
        raise ValueError(f"Empty artifacts: {empty}")

    summary = json.loads((run_dir / "summary.json").read_text())
    if tuple(summary["groups"]) != EXPECTED_GROUPS:
        raise ValueError(f"Unexpected groups: {tuple(summary['groups'])}")
    if len(summary["t_steps"]) != EXPECTED_STATES:
        raise ValueError(f"Expected {EXPECTED_STATES} states")
    for group, rows in summary["groups"].items():
        if len(rows) != EXPECTED_STATES:
            raise ValueError(f"{group} has {len(rows)} states")
        if any(row["samples"] != EXPECTED_SAMPLES for row in rows):
            raise ValueError(f"{group} does not have {EXPECTED_SAMPLES} samples/state")

    line_count = sum(1 for _ in (run_dir / "per_sample.jsonl").open())
    expected_lines = len(EXPECTED_GROUPS) * EXPECTED_SAMPLES * EXPECTED_STATES
    if line_count != expected_lines:
        raise ValueError(f"Expected {expected_lines} sample rows, found {line_count}")

    payload = torch.load(run_dir / "trajectory_tensors.pt", map_location="cpu", mmap=True)
    expected_z = (120, EXPECTED_STATES, 132, 512)
    expected_v = (120, EXPECTED_STEPS, 132, 512)
    if tuple(payload["z"].shape) != expected_z:
        raise ValueError(f"Unexpected z shape: {tuple(payload['z'].shape)}")
    if tuple(payload["velocity"].shape) != expected_v:
        raise ValueError(f"Unexpected velocity shape: {tuple(payload['velocity'].shape)}")
    if len(set(payload["source_ids"].tolist())) != 120:
        raise ValueError("Source IDs are not 120 unique examples")

    findings = {}
    for group, rows in summary["groups"].items():
        valid_velocity = [row for row in rows if row["velocity_norm"] is not None]
        valid_cosine = [
            row for row in rows if row["velocity_cosine_previous"] is not None
        ]
        findings[group] = {
            "initial_z_accuracy": rows[0]["z_accuracy"],
            "final_z_accuracy": rows[-1]["z_accuracy"],
            "final_xpred_accuracy": rows[-1]["xpred_accuracy"],
            "z_auc": trapezoid_auc(rows, "z_accuracy"),
            "xpred_auc": trapezoid_auc(rows, "xpred_accuracy"),
            "z_crossing_50": first_crossing(rows, "z_accuracy", 0.50),
            "z_crossing_80": first_crossing(rows, "z_accuracy", 0.80),
            "z_crossing_90": first_crossing(rows, "z_accuracy", 0.90),
            "xpred_crossing_90": first_crossing(rows, "xpred_accuracy", 0.90),
            "peak_velocity": max(valid_velocity, key=lambda row: row["velocity_norm"]),
            "minimum_velocity_cosine": min(
                valid_cosine, key=lambda row: row["velocity_cosine_previous"]
            ),
        }

    report = {
        "verified": True,
        "run_dir": str(run_dir),
        "checkpoint": summary["checkpoint"],
        "checkpoint_step": summary["checkpoint_step"],
        "samples": 120,
        "states": EXPECTED_STATES,
        "tensor_shapes": {"z": list(expected_z), "velocity": list(expected_v)},
        "findings": findings,
    }
    (run_dir / "verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    print(json.dumps(validate(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
