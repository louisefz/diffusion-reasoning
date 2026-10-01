#!/bin/bash
#SBATCH --job-name=sudoku-200k-feedback
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --chdir=/data/leuven/377/vsc37788/continuous-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=sudoku-200k-feedback-%j.out

set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export XDG_CACHE_HOME=/tmp/cache-${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false

/data/leuven/377/vsc37788/envs/elf-official/bin/python -u official_feedback_ablation.py \
  --config official-configs/sudoku9_capability.yml \
  --checkpoint runs/official-sudoku9-capability-62169683/checkpoint_200000 \
  --output-dir "runs/official-sudoku-200k-hard-feedback-${SLURM_JOB_ID}" \
  --difficulty hard --num-samples 256 --batch-size 1 --restarts 4 \
  --flow-steps 16 --time-schedule uniform --cfg 1 --seed 20260930
