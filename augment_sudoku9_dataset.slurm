#!/bin/bash
#SBATCH --job-name=sudoku-augment
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=sudoku-augment-%j.out

set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

/data/leuven/377/vsc37788/envs/elf-official/bin/python -u augment_sudoku9_dataset.py \
  --source data/official-sudoku9-difficulty-v1 \
  --output data/official-sudoku9-symmetry-300k-v1 \
  --tokenizer cache/models/t5-small \
  --train-examples 300000
