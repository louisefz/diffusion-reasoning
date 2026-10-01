import random
import unittest

from graph_task import episode, generate, shortest_distances, validate
from make_official_graph import render


class GraphTaskTest(unittest.TestCase):
    def test_controlled_positive_depth_and_matched_lookup(self):
        reach, lookup = episode(
            random.Random(9), 8, 4, True, False, set(), "test", 0,
        )
        edges = {tuple(edge) for edge in reach["edges"]}
        distance = shortest_distances(8, edges, reach["source"])[reach["target"]]
        self.assertEqual(distance, 4)
        self.assertEqual(reach["states"], [0, 0, 0, 0, 1])
        self.assertEqual(lookup["answer"], 0)
        self.assertEqual(reach["graph_id"], lookup["graph_id"])

    def test_balanced_generation_and_validation(self):
        rows = generate(11, 8, (1, 2, 4), 24, set(), "validation")
        validate(rows, 8)
        for task in ("reach", "lookup"):
            for depth in (1, 2, 4):
                labels = [row["answer"] for row in rows
                          if row["task"] == task and row["depth"] == depth]
                self.assertEqual(labels.count(0), labels.count(1))

    def test_render_is_unambiguous(self):
        reach, lookup = episode(
            random.Random(13), 8, 2, False, True, set(), "test", 0,
        )
        self.assertIn("Mode REACH", render(reach, 8))
        self.assertIn("Mode EDGE", render(lookup, 8))
        self.assertIn("Answer 0 or 1:", render(reach, 8))


if __name__ == "__main__":
    unittest.main()
