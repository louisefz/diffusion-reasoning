import sys
import unittest
from pathlib import Path

OFFICIAL_SRC = Path(__file__).parents[1] / "official-elf" / "src"
sys.path.insert(0, str(OFFICIAL_SRC))
from generation import answer_accuracy_metrics


class OfficialMetricTests(unittest.TestCase):
    def test_exact_and_first_integer(self):
        result = answer_accuracy_metrics(["7", "answer: 3 extra", "8"], ["7", "3", "2"])
        self.assertAlmostEqual(result["exact_match"], 100 / 3)
        self.assertAlmostEqual(result["first_integer_accuracy"], 200 / 3)
