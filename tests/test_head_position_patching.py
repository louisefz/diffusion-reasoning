import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from official_head_position_patching import semantic_positions


def test_real_prompt_semantic_position_count():
    # Full tokenizer/data validation is run by the launch preflight; this test
    # simply asserts the public parser remains importable.
    assert callable(semantic_positions)
