#!/bin/bash
#SBATCH --job-name=sudoku-flow-correction
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --chdir=/data/leuven/377/vsc37788/continuous-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=56G
#SBATCH --time=02:00:00
#SBATCH --output=sudoku-flow-correction-%j.out

set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/triton-${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-${SLURM_JOB_ID}
PY=/data/leuven/377/vsc37788/envs/elf-official/bin/python

OUT="runs/official-sudoku-easy-flow-correction-${SLURM_JOB_ID}"
"$PY" -u official_flow_error_correction.py \
  --config official-configs/sudoku9_capability.yml \
  --checkpoint runs/official-sudoku9-capability-62169683/checkpoint_100000 \
  --output-dir "$OUT" --difficulty easy \
  --num-samples 96 --batch-size 1 --restarts 8 --flow-steps 16 \
  --time-schedule uniform --cfg 1
