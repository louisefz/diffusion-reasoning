import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from official_branching_stability import (
    _answer_agreement, _generated_rms, aggregate_rows,
)


def test_answer_agreement_handles_variable_length_spans():
    predictions = torch.tensor([[3, 6, 9], [2, 8, 9]])
    references = torch.tensor([[3, 6, 0], [2, 7, 9]])
    result = _answer_agreement(
        predictions, references, torch.tensor([0, 0]), torch.tensor([2, 1]),
    )
    assert result.tolist() == [True, True]


def test_generated_rms_ignores_condition_slots():
    z = torch.tensor([[[100.0], [3.0], [4.0]]])
    cond_mask = torch.tensor([[1.0, 0.0, 0.0]])
    assert torch.allclose(_generated_rms(z, cond_mask), torch.tensor([3.535534]))


def test_aggregate_uses_example_level_means():
    rows = []
    for source_id, outcomes in ((1, [True, True]), (2, [False, False])):
        for branch, correct in enumerate(outcomes):
            rows.append({
                "group": "compose/d1", "t_index": 2, "t_state": 0.1,
                "sigma": 0.1, "perturb_target": "z", "source_id": source_id,
                "baseline_correct": True,
                "correct": correct, "agrees_with_baseline": correct,
                "z_noise_rms": 0.1, "selfcond_noise_rms": 0.0,
                "branch": branch,
            })
    result = aggregate_rows(rows)[0]
    assert result["samples"] == 2
    assert result["rollouts"] == 4
    assert result["correct_rate"] == 0.5
    assert result["retention_rate_given_baseline_correct"] == 0.5
