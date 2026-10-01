#!/bin/bash
#SBATCH --job-name=sudoku9-flow-sweep
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=sudoku9-flow-sweep-%A_%a.out

set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export XDG_CACHE_HOME=/tmp/cache-${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false

FLOW_STEPS=(32 50)
M=${FLOW_STEPS[$SLURM_ARRAY_TASK_ID]}

/data/leuven/377/vsc37788/envs/elf-official/bin/python -u official_sudoku_transition_eval.py \
  --config official-configs/sudoku9_transition.yml \
  --checkpoint runs/official-sudoku9-transition-62187260/checkpoint_100000 \
  --data-path data/official-sudoku9-transition-220k-v1/test \
  --output-dir "runs/official-sudoku9-flow${M}-eval-${SLURM_JOB_ID}" \
  --num-one-step 64 --one-step-restarts 2 \
  --num-rollout-puzzles 0 --rollout-restarts 1 \
  --max-reasoning-steps 1 --flow-steps "$M" --batch-size 8 --seed 20260930
