import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from official_intermediate_state_probe import _first_layer


def test_state_specific_first_layer():
    rows = [
        {"state_step": 1, "layer": 0, "eval_accuracy": 0.2},
        {"state_step": 1, "layer": 1, "eval_accuracy": 0.9},
        {"state_step": 2, "layer": 0, "eval_accuracy": 0.1},
        {"state_step": 2, "layer": 1, "eval_accuracy": 0.4},
    ]
    assert _first_layer(rows, 1, 0.8) == 1
    assert _first_layer(rows, 2, 0.8) is None
