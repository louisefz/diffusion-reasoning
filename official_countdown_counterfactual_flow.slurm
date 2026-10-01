#!/bin/bash
#SBATCH --job-name=countdown-cf-flow
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100_debug
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:a100_80gb_pcie:1
#SBATCH --chdir=/data/leuven/377/vsc37788/continuous-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --output=countdown-cf-flow-%j.out

set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/triton-${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-${SLURM_JOB_ID}

OUT="runs/official-countdown-counterfactual-flow-${SLURM_JOB_ID}"
/data/leuven/377/vsc37788/envs/elf-official/bin/python -u \
  official_countdown_counterfactual_flow.py \
  --config official-configs/countdown_capability.yml \
  --checkpoint runs/official-countdown-capability-62169682/checkpoint_50000 \
  --output-dir "$OUT" \
  --depth 1 --depth 2 --depth 3 --depth 4 \
  --pairs-per-depth 75 \
  --batch-size 16 \
  --flow-steps 16 \
  --cfg 1
