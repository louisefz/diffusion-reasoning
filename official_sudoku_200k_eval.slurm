#!/bin/bash
#SBATCH --job-name=sudoku-200k-eval
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --chdir=/data/leuven/377/vsc37788/continuous-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=sudoku-200k-eval-%j.out

set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

OUT="runs/official-sudoku-200k-eval-${SLURM_JOB_ID}.json"
/data/leuven/377/vsc37788/envs/elf-official/bin/python -u official_sudoku_eval.py \
  --config official-configs/sudoku9_capability.yml \
  --checkpoint runs/official-sudoku9-capability-62169683/checkpoint_200000 \
  --output "$OUT" --num-samples 1500 --batch-size 32 \
  --steps 16 32 64 --cfgs 1 2
