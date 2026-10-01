# Diffusion Reasoning

Research code for studying reasoning dynamics in continuous diffusion and
flow-matching language models. This repository contains controlled computation
tasks, an adapted PyTorch ELF implementation, causal diagnostics, recurrent
self-conditioning experiments, and verifier-executable Sudoku action flows.

The central question is not merely whether a diffusion model predicts the
right answer, but whether its iterative dynamics implement reusable
computation:

```text
problem state -> continuous flow -> executable transition -> next state
```

## Main experiment families

### Structured Sudoku action flow

`structured_sudoku_action_flow.py` compares two parameter-matched models:

- a direct row/column/value classifier;
- a continuous action flow that transports Gaussian noise to three categorical
  action vertices.

Actions are executed by a Sudoku verifier, so evaluation separates target
matching from action legality. Relevant entry points:

- `make_sudoku_transition_data.py`
- `make_sudoku_action_data.py`
- `pack_structured_sudoku_action.py`
- `structured_sudoku_action_flow.py`
- `structured_sudoku_action_overfit.slurm`
- `structured_sudoku_action_full.slurm`

The overfit experiment is an implementation gate. It is not evidence of rule
generalization. Full experiments evaluate held-out legal-action rate and are
followed by autonomous rollouts.

### Repeatable computation transitions

- `computation_transition_bfs.py` separates outer algorithmic depth `K` from
  inner flow integration NFE `M` on controlled graph reachability.
- `computation_transition_multipath.py` studies stochastic flows on tasks with
  multiple valid computation paths and compares against a stochastic recurrent
  operator.

### ELF reasoning and causal diagnostics

`official-elf/` is a vendored, locally modified PyTorch ELF implementation
based on [lillian039/ELF](https://github.com/lillian039/ELF). Its upstream
license is preserved in `official-elf/LICENSE`.

The `official_*.py` scripts cover trajectory decoding, counterfactual patching,
layer/head localization, self-conditioning ablations, error injection,
constraint relaxation, compute allocation, and transition/action evaluation.
Task configurations are in `official-configs/`.

## Installation

Use Python 3.10+ and a recent PyTorch build. For the adapted ELF code:

```bash
python -m pip install -r official-elf/requirements.txt
python -m pip install -r requirements-wandb.txt
```

Cluster paths and Slurm accounts in `*.slurm` and `official-configs/*.yml` are
site-specific examples and must be changed for a new environment.

## Minimal controlled-task test

```bash
python make_data.py --out data/pilot --train-episodes 10000 --eval-episodes 500
python -m unittest discover -s tests -v
python train.py --data data/pilot --out runs/pilot --steps 20000 --device cuda \
  --data-mode online
```

## Reproducibility and interpretation

- Data, checkpoints, W&B state, scheduler logs, and authentication files are
  intentionally excluded from version control.
- Increasing numerical NFE is not automatically an increase in reasoning
  depth.
- Decodability, causal use, perturbation stability, and executable computation
  are evaluated separately.
- Synthetic-task success is not presented as evidence of general language
  reasoning without replication on a language model.

`STATUS.md` contains the chronological experimental record. The reports in the
repository summarize selected mechanistic analyses.

## Upstream

ELF: *Embedded Language Flows*,
[arXiv:2605.10938](https://arxiv.org/abs/2605.10938).
