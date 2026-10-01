import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from official_counterfactual_patching import counterfactual_for_step, execute, parse_tables


class TinyTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(map(ord, text))}


def test_minimal_counterfactual_preserves_prefix_and_changes_answer():
    tables = [list(range(8)), [1, 2, 3, 4, 5, 6, 7, 0],
              [2, 3, 4, 5, 6, 7, 0, 1], [3, 4, 5, 6, 7, 0, 1, 2]]
    program = [0, 1, 2, 3]
    states = execute(tables, 0, program)
    prompt = (
        "Each function list gives outputs for inputs 0 through 7. "
        + "; ".join(f"F{i}: " + " ".join(map(str, table)) for i, table in enumerate(tables))
        + ". Mode FULL. Start 0. Program F0 F1 F2 F3. "
          "FULL applies every function; FIRST applies only the first function. Answer:"
    )
    row = {"input": prompt, "program": program, "states": states,
           "episode_id": "x", "table_id": "y"}
    cf = counterfactual_for_step(row, 3, TinyTokenizer(), random.Random(0))
    assert cf is not None
    assert cf["states"][:3] == states[:3]
    assert cf["states"][3] != states[3]
    assert cf["states"][-1] != states[-1]
    assert sorted(parse_tables(cf["input"])[program[2]]) == list(range(8))
