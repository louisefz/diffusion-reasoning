#!/bin/bash
#SBATCH --job-name=sudoku9-action-data
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=sudoku9-action-data-%j.out

set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

/data/leuven/377/vsc37788/envs/elf-official/bin/python -u make_sudoku_action_data.py \
  --source data/official-sudoku9-transition-220k-v1 \
  --output data/official-sudoku9-action-220k-v1 \
  --tokenizer cache/models/t5-small --num-proc 8
