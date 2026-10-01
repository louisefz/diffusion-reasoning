import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from official_component_patching import aggregate


def test_component_aggregate():
    rows = [
        {"group": "g", "donor_type": "different", "block": 10,
         "component": "mlp", "follows_source": True, "follows_donor": False,
         "follows_other": False, "exact_correct": True},
        {"group": "g", "donor_type": "different", "block": 10,
         "component": "mlp", "follows_source": False, "follows_donor": True,
         "follows_other": False, "exact_correct": False},
    ]
    result = aggregate(rows)[0]
    assert result["samples"] == 2
    assert result["source_answer_rate"] == 0.5
    assert result["donor_answer_rate"] == 0.5
