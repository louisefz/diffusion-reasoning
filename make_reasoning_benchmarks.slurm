#!/bin/bash
#SBATCH --job-name=make-reason-data
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --chdir=/data/leuven/377/vsc37788/continuous-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=make-reason-data-%A_%a.out
#SBATCH --array=0-2

set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY=/data/leuven/377/vsc37788/envs/elf-official/bin/python
TOKENIZER=cache/models/t5-small

if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
  "$PY" -u make_official_graph.py \
    --out data/official-graph-depth-v1 --tokenizer "$TOKENIZER" --nodes 16 \
    --train-depths 1,2,3,4,5 --eval-depths 1,2,3,4,5,6,8,10,12 \
    --train-episodes 200000 --eval-episodes 1800 --seed 20260927
elif [[ "${SLURM_ARRAY_TASK_ID}" == "1" ]]; then
  "$PY" -u make_official_standard_reasoning.py \
    --task countdown --out data/official-countdown-depth-v1 --tokenizer "$TOKENIZER" \
    --train-examples 100000 --eval-examples 1400 --seed 20260927
else
  "$PY" -u make_official_standard_reasoning.py \
    --task sudoku --out data/official-sudoku9-difficulty-v1 --tokenizer "$TOKENIZER" \
    --train-examples 30000 --eval-examples 1500 --seed 20260927
fi
