import unittest
from task import OnlineEpisodes, encode, generate, validate


class OnlineTests(unittest.TestCase):
    def test_exclusion_even_when_rng_matches_holdout(self):
        holdout = generate(42, 8, 4, [1, 2, 4, 8], 16, set(), "test")
        keys = {r["table_id"] for r in holdout}
        stream = OnlineEpisodes(42, 8, 4, [1, 2, 4, 8], keys)
        seen = set()
        for _ in range(10):
            rows = stream.next_batch(16)
            validate(rows, 8, 4)
            new = {r["table_id"] for r in rows}
            self.assertEqual(len(new), 8)
            self.assertFalse(new & (keys | seen))
            seen.update(new)
            pairs = {}
            for r in rows:
                pairs.setdefault(r["episode_id"], []).append(r)
            for pair in pairs.values():
                self.assertEqual({r["task"] for r in pair}, {"compose", "lookup"})
                self.assertEqual(encode(pair[0], 8, 4)[1:], encode(pair[1], 8, 4)[1:])

    def test_resume_is_exact_with_different_initial_seed(self):
        stream = OnlineEpisodes(7, 8, 4, [1, 2, 4, 8], set())
        stream.next_batch(10)
        saved = stream.state_dict()
        expected = stream.next_batch(10)
        resumed = OnlineEpisodes(999, 8, 4, [1, 2, 4, 8], set())
        resumed.load_state_dict(saved)
        self.assertEqual(resumed.next_batch(10), expected)

    def test_invalid_batch_and_incompatible_resume(self):
        stream = OnlineEpisodes(7, 8, 4, [1], set())
        with self.assertRaises(ValueError):
            stream.next_batch(3)
        state = stream.state_dict()
        other = OnlineEpisodes(7, 8, 4, [2], set())
        with self.assertRaises(ValueError):
            other.load_state_dict(state)


if __name__ == "__main__":
    unittest.main()
