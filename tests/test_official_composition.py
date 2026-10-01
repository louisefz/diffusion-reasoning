import unittest

from make_official_composition import render
from task import episode


class OfficialCompositionDataTest(unittest.TestCase):
    def test_paired_prompts_differ_only_in_mode_and_have_correct_answers(self):
        import random
        compose, lookup = episode(random.Random(7), 8, 4, 4, set(), "test", 0)
        compose_prompt = render(compose)
        lookup_prompt = render(lookup)
        self.assertIn("Mode FULL", compose_prompt)
        self.assertIn("Mode FIRST", lookup_prompt)
        self.assertEqual(compose_prompt.replace("FULL", "FIRST", 1), lookup_prompt)
        self.assertEqual(compose["answer"], compose["states"][-1])
        self.assertEqual(lookup["answer"], compose["states"][1])


if __name__ == "__main__":
    unittest.main()
