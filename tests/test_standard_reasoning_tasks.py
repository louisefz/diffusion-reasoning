import random
import unittest

from countdown_task import episode as countdown_episode, verify as verify_countdown
from sudoku_task import episode as sudoku_episode, solutions, verify as verify_sudoku
from sudoku9_task import episode as sudoku9_episode, solve_with_metrics, verify as verify_sudoku9


class StandardReasoningTaskTest(unittest.TestCase):
    def test_countdown_certified_depth_and_verifier(self):
        for depth in (1, 2, 3, 4):
            row = countdown_episode(random.Random(100 + depth), depth)
            self.assertTrue(verify_countdown(row))

    def test_unique_sudoku_and_verifier(self):
        for blanks in (4, 6, 8):
            row = sudoku_episode(random.Random(200 + blanks), blanks)
            self.assertEqual(len(solutions(row["puzzle"], 2)), 1)
            self.assertTrue(verify_sudoku(row))

    def test_sudoku9_solver_metrics_and_verifier(self):
        row = sudoku9_episode(random.Random(309), 30)
        found, metrics = solve_with_metrics(row["puzzle"], 2)
        self.assertEqual(len(found), 1)
        self.assertTrue(verify_sudoku9(row))
        self.assertIn("propagation_rounds", metrics)


if __name__ == "__main__":
    unittest.main()
