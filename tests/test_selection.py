import unittest
from task import OnlineEpisodes
from train import training_rows


class SelectionTests(unittest.TestCase):
    def test_lookup_batch_and_resume(self):
        stream = OnlineEpisodes(0, 8, 4, [1, 2, 4, 8], set())
        rows = training_rows(stream, None, 8, "lookup")
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(r['task'] == 'lookup' for r in rows))
        self.assertEqual(stream.count, 8)
        state = stream.state_dict()
        expected = training_rows(stream, None, 8, "lookup")
        other = OnlineEpisodes(1, 8, 4, [1, 2, 4, 8], set())
        other.load_state_dict(state)
        self.assertEqual(expected, training_rows(other, None, 8, "lookup"))
