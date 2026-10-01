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
#SBATCH --output=sudoku-transition-data-%j.out

set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

/data/leuven/377/vsc37788/envs/elf-official/bin/python -u make_sudoku_transition_data.py \
  --output data/official-sudoku-transition-joint-220k-v1 \
  --tokenizer cache/models/t5-small \
  --train-examples 200000 \
  --validation-examples 10000 \
  --test-examples 10000
