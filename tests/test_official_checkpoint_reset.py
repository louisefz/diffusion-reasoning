import sys
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "official-elf" / "src"))

from utils.checkpoint_utils import load_checkpoint
from utils.train_utils import TrainState


def test_load_checkpoint_can_reset_optimizer_state_and_keep_counters():
    source = torch.nn.Linear(3, 2)
    source_optimizer = torch.optim.AdamW(source.parameters(), lr=0.0123)
    source(torch.ones(1, 3)).sum().backward()
    source_optimizer.step()

    payload = {
        "params": source.state_dict(),
        "ema_params1": {
            name: value.detach().clone() for name, value in source.named_parameters()
        },
        "opt_state": source_optimizer.state_dict(),
        "lr_scheduler": None,
        "step": 76024,
        "epoch": 7,
        "dropout_rng": None,
        "grad_accum_buffers": {},
    }

    target = torch.nn.Linear(3, 2)
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=0.5)
    state = TrainState(model=target, optimizer=target_optimizer)

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "checkpoint_76024"
        torch.save(payload, checkpoint)
        state, step = load_checkpoint(
            str(checkpoint), state, restore_optimizer=False
        )

    assert step == 76024
    assert state.step == 76024
    assert state.epoch == 7
    assert state.optimizer.state == {}
    assert state.optimizer.param_groups[0]["lr"] == 0.0123
    for source_value, target_value in zip(source.parameters(), target.parameters()):
        assert torch.equal(source_value, target_value)
