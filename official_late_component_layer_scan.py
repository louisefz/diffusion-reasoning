#!/usr/bin/env python
"""Scan every ELF block for individually bypassable late attention/MLP updates."""

import official_late_component_bypass as experiment


experiment.BLOCK_SETS = {f"b{block}": (block,) for block in range(1, 13)}
experiment.MODE_SPECS = {
    f"all_b{block}_{component}": ("all", (block,), component)
    for block in range(1, 13)
    for component in ("attn", "mlp")
}


if __name__ == "__main__":
    experiment.run(experiment.parse_args())
