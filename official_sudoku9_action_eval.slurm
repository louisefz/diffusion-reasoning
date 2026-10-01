#!/bin/bash
#SBATCH --job-name=sudoku9-action-eval
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=sudoku9-action-eval-%j.out

set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export XDG_CACHE_HOME=/tmp/cache-${SLURM_JOB_ID}
export TRITON_CACHE_DIR=/tmp/triton-${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false

TRAIN_JOB=${TRAIN_JOB:?Set TRAIN_JOB to the action pilot training job id}
CHECKPOINT=/data/leuven/377/vsc37788/continuous-reasoning/runs/official-sudoku9-action-pilot-${TRAIN_JOB}/checkpoint_20000
OUTPUT=/data/leuven/377/vsc37788/continuous-reasoning/results/sudoku9-action-pilot-${TRAIN_JOB}

/data/leuven/377/vsc37788/envs/elf-official/bin/python -u official_sudoku_action_eval.py \
  --config official-configs/sudoku9_action_pilot.yml \
  --checkpoint "$CHECKPOINT" \
  --data-path data/official-sudoku9-action-220k-v1/test \
  --output-dir "$OUTPUT" \
  --num-one-step 256 \
  --one-step-restarts 4 \
  --num-rollout-puzzles 64 \
  --rollout-restarts 4 \
  --max-reasoning-steps 32 \
  --flow-steps 32 \
  --batch-size 32
