import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

OFFICIAL_SRC = Path(__file__).parents[1] / "official-elf" / "src"
sys.path.insert(0, str(OFFICIAL_SRC))
from utils.checkpoint_utils import initialize_model_weights


class OfficialInitTests(unittest.TestCase):
    def test_weights_only_prefers_ema_and_resets_counters(self):
        source = torch.nn.Linear(3, 2)
        target = torch.nn.Linear(3, 2)
        ema = {name: torch.full_like(value, 7) for name, value in source.named_parameters()}
        payload = {"params": source.state_dict(), "ema_params1": ema,
                   "opt_state": {"must": "not load"}, "step": 99, "epoch": 4}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint_99"
            torch.save(payload, path)
            state = SimpleNamespace(model=target, ema_params1={}, step=8, epoch=3)
            initialize_model_weights(str(path), state, use_ema=True)
        for value in target.parameters():
            self.assertTrue(torch.equal(value, torch.full_like(value, 7)))
        self.assertEqual((state.step, state.epoch), (0, 0))
