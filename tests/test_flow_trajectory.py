import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from official_flow_trajectory import (
    _empty_group_rows, _finalize, answer_decoder_stats, parse_group_spec,
    select_balanced_indices,
)


def test_parse_group_spec():
    assert parse_group_spec("compose:4:30") == ("compose", 4, 30)


def test_balanced_selection_is_deterministic_and_grouped():
    dataset = [
        {"task": task, "depth": depth, "target": str(i % 8), "input": str(i)}
        for task, depth in (("compose", 1), ("compose", 2), ("lookup", 4))
        for i in range(10)
    ]
    specs = ["compose:1:3", "compose:2:2", "lookup:4:4"]
    indices1, metadata1 = select_balanced_indices(dataset, specs, seed=7)
    indices2, metadata2 = select_balanced_indices(dataset, specs, seed=7)
    assert indices1 == indices2
    assert metadata1 == metadata2
    assert [row["group"] for row in metadata1] == (
        ["compose/d1"] * 3 + ["compose/d2"] * 2 + ["lookup/d4"] * 4
    )


def test_answer_decoder_stats_uses_each_samples_answer_position():
    logits = torch.zeros(2, 4, 5)
    answer_positions = torch.tensor([1, 3])
    answer_lengths = torch.tensor([1, 1])
    targets = torch.tensor([[0, 2, 0, 0], [0, 0, 0, 4]])
    logits[0, 1, 2] = 5.0
    logits[0, 1, 1] = 2.0
    logits[1, 3, 3] = 6.0
    logits[1, 3, 4] = 1.0
    stats = answer_decoder_stats(
        logits, answer_positions, answer_lengths, targets,
    )
    assert stats["prediction"].tolist() == [2, 3]
    assert stats["target"].tolist() == [2, 4]
    assert stats["correct"].tolist() == [True, False]
    assert torch.allclose(stats["target_margin"], torch.tensor([3.0, -5.0]))


def test_answer_decoder_stats_handles_multitoken_zero():
    logits = torch.zeros(1, 4, 10)
    targets = torch.tensor([[3, 6, 1, 1]])
    logits[0, 0, 3] = 4.0
    logits[0, 1, 6] = 5.0
    stats = answer_decoder_stats(
        logits, torch.tensor([0]), torch.tensor([2]), targets,
    )
    assert stats["correct"].tolist() == [True]
    assert stats["target"].tolist() == [6]


def test_finalize_removes_empty_initial_state_accumulators():
    result = _finalize(_empty_group_rows(["compose/d1"], [0.0]))
    row = result["compose/d1"][0]
    assert row["xpred_accuracy"] is None
    assert row["velocity_norm"] is None
    assert row["velocity_cosine_previous"] is None
    assert not any(key.endswith("_sum") for key in row)
