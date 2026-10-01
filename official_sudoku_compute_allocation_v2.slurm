#!/bin/bash
#SBATCH --job-name=sudoku-allocate-9call
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
#SBATCH --output=sudoku-allocate-9call-%j.out

set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export XDG_CACHE_HOME=/tmp/cache-${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false

/data/leuven/377/vsc37788/envs/elf-official/bin/python -u official_sudoku_compute_allocation_v2.py \
  --config official-configs/sudoku9_capability.yml \
  --checkpoint runs/official-sudoku9-capability-62169683/checkpoint_200000 \
  --output-dir "runs/official-sudoku-compute-allocation-v2-${SLURM_JOB_ID}" \
  --difficulty hard --num-samples 500 --batch-size 4 --restarts 1 \
  --cfg 1 --seed 20261005
