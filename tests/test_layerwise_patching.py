import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from official_layerwise_patching import donor_indices


def test_donor_indices_respect_answer_relation():
    labels = [0, 0, 1, 1, 2, 2]
    same = donor_indices(labels, same_answer=True, seed=3).tolist()
    different = donor_indices(labels, same_answer=False, seed=3).tolist()
    for row in range(len(labels)):
        assert same[row] != row and labels[same[row]] == labels[row]
        assert different[row] != row and labels[different[row]] != labels[row]
