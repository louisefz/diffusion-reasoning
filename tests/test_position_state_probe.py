import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from official_position_state_probe import ROLE_NAMES, semantic_role_indices


class TinyTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        # Character tokenizer makes expected semantic spans unambiguous.
        result = {"input_ids": list(map(ord, text))}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result


def test_semantic_roles_match_ground_truth_path():
    prompt = (
        "Each function list gives outputs for inputs 0 through 7. "
        "F0: 0 1 2 3 4 5 6 7; F1: 1 2 3 4 5 6 7 0; "
        "F2: 2 3 4 5 6 7 0 1; F3: 3 4 5 6 7 0 1 2. "
        "Mode FULL. Start 2. Program F1 F2 F3 F0. "
        "FULL applies every function; FIRST applies only the first function. Answer:"
    )
    row = {
        "input": prompt, "condition_input_ids": list(map(ord, prompt)),
        "program": [1, 2, 3, 0], "states": [2, 3, 5, 0, 0],
    }
    roles = semantic_role_indices(row, TinyTokenizer())
    assert set(roles) == set(ROLE_NAMES) - {"answer"}
    assert "".join(prompt[i] for i in roles["start"]) == "2"
    assert "".join(prompt[i] for i in roles["program_2"]) == "F2"
    assert "".join(prompt[i] for i in roles["oracle_cell_1"]) == "3"
    assert "".join(prompt[i] for i in roles["oracle_cell_2"]) == "5"
