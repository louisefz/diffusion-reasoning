#!/bin/bash
#SBATCH --job-name=sudoku-200k-dynamics
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
#SBATCH --array=0-2
#SBATCH --output=sudoku-200k-dynamics-%A_%a.out

set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export XDG_CACHE_HOME=/tmp/cache-${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false

case "${SLURM_ARRAY_TASK_ID}" in
  0) DIFFICULTY=easy ;;
  1) DIFFICULTY=medium ;;
  2) DIFFICULTY=hard ;;
esac

/data/leuven/377/vsc37788/envs/elf-official/bin/python -u official_autonomous_correction.py \
  --config official-configs/sudoku9_capability.yml \
  --checkpoint runs/official-sudoku9-capability-62169683/checkpoint_200000 \
  --output-dir "runs/official-sudoku-200k-${DIFFICULTY}-dynamics-${SLURM_JOB_ID}" \
  --difficulty "${DIFFICULTY}" --num-samples 128 --batch-size 1 --restarts 4 \
  --flow-steps 16 --time-schedule uniform --cfg 1 --seed 20260930
