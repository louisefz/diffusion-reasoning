import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from official_head_patching import aggregate


def test_head_aggregate():
    rows = [
        {"intervention_step": 1, "mode": "individual", "head": 0,
         "follows_counterfactual": True, "follows_original": False, "follows_other": False,
         "original_baseline_correct": True, "counterfactual_baseline_correct": True},
        {"intervention_step": 1, "mode": "individual", "head": 0,
         "follows_counterfactual": False, "follows_original": True, "follows_other": False,
         "original_baseline_correct": True, "counterfactual_baseline_correct": True},
    ]
    result = aggregate(rows)[0]
    assert result["samples"] == 2
    assert result["counterfactual_answer_rate"] == 0.5
