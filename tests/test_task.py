import copy
import unittest
from task import encode, execute, generate, validate


class TaskTests(unittest.TestCase):
    def test_known_composition(self):
        self.assertEqual(execute([[1, 2, 0], [2, 1, 0]], 0, [0, 1, 0]), [0, 1, 1, 2])

    def test_splits_and_permutations(self):
        seen = set()
        a = generate(1, 8, 4, [1, 2, 4, 8], 100, seen, "a")
        b = generate(2, 8, 4, [1, 2, 12, 16], 100, seen, "b")
        validate(a, 8, 4)
        validate(b, 8, 4)
        self.assertFalse({r["table_id"] for r in a} & {r["table_id"] for r in b})
        self.assertEqual(a, generate(1, 8, 4, [1, 2, 4, 8], 100, set(), "a"))

    def test_no_answer_or_state_leakage(self):
        row = generate(1, 8, 4, [4], 1, set(), "a")[0]
        changed = copy.deepcopy(row)
        changed.update(answer=99, states=[99]*10)
        self.assertEqual(encode(row, 8, 4), encode(changed, 8, 4))

    def test_paired_controls(self):
        rows = generate(1, 8, 4, [1, 2, 4], 12, set(), "a")
        by_id = {}
        for r in rows:
            by_id.setdefault(r["episode_id"], []).append(r)
        for pair in by_id.values():
            a, b = [encode(r, 8, 4) for r in pair]
            self.assertNotEqual(a[0], b[0])
            self.assertEqual(a[1:], b[1:])


if __name__ == "__main__":
    unittest.main()

