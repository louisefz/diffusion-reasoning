#!/bin/bash
#SBATCH --job-name=sudoku-transition-data
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=sudoku-transition-data-%A_%a.out

set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

SIZES=(4 9)
SIZE="${SIZES[$SLURM_ARRAY_TASK_ID]}"

/data/leuven/377/vsc37788/envs/elf-official/bin/python -u make_sudoku_transition_data.py \
  --output "data/official-sudoku${SIZE}-transition-220k-v1" \
  --tokenizer cache/models/t5-small \
  --sizes "$SIZE" \
  --train-examples 200000 \
  --validation-examples 10000 \
  --test-examples 10000 \
  --seed "$((20260929 + SIZE * 1000))"
