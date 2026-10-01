#!/bin/bash
#SBATCH --job-name=structured-action-full
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=structured-action-full-%j.out

set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

OUTPUT=/data/leuven/377/vsc37788/continuous-reasoning/results/structured-action-full-${SLURM_JOB_ID}
/data/leuven/377/vsc37788/envs/elf-official/bin/python -u structured_sudoku_action_flow.py \
  --train-data data/official-sudoku9-action-220k-v1/train \
  --test-data data/official-sudoku9-action-220k-v1/test \
  --output-dir "$OUTPUT" \
  --train-limit 200000 \
  --test-limit 10000 \
  --steps 20000 \
  --batch-size 256 \
  --eval-batch-size 512 \
  --evaluation-limit 4096 \
  --verifier-limit 1024 \
  --eval-every 2000 \
  --nfe 16
