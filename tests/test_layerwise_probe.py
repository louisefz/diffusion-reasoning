import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from official_layerwise_probe import _first_layer


def test_first_layer_threshold():
    metrics = [
        {"layer": 0, "score": 0.2},
        {"layer": 1, "score": 0.6},
        {"layer": 2, "score": 0.9},
    ]
    assert _first_layer(metrics, "score", 0.5) == 1
    assert _first_layer(metrics, "score", 0.9) == 2
    assert _first_layer(metrics, "score", 0.95) is None
