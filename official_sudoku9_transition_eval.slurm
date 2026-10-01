#!/bin/bash
#SBATCH --job-name=sudoku9-transition-eval
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=sudoku9-transition-eval-%j.out

set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export XDG_CACHE_HOME=/tmp/cache-${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false

/data/leuven/377/vsc37788/envs/elf-official/bin/python -u official_sudoku_transition_eval.py \
  --config official-configs/sudoku9_transition.yml \
  --checkpoint runs/official-sudoku9-transition-62187260/checkpoint_100000 \
  --data-path data/official-sudoku9-transition-220k-v1/test \
  --output-dir "runs/official-sudoku9-transition-eval-${SLURM_JOB_ID}" \
  --num-one-step 128 --one-step-restarts 4 \
  --num-rollout-puzzles 32 --rollout-restarts 4 \
  --max-reasoning-steps 16 --flow-steps 16 --batch-size 8 --seed 20260930
