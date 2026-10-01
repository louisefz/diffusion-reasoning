#!/bin/bash
#SBATCH --job-name=sudoku-scaling
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
#SBATCH --output=sudoku-scaling-%j.out

set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
PY=/data/leuven/377/vsc37788/envs/elf-official/bin/python
RUN=runs/official-sudoku9-capability-62169683

for STEP in 18750 37500 50000; do
  "$PY" -u official_sudoku_eval.py \
    --config official-configs/sudoku9_capability.yml \
    --checkpoint "$RUN/checkpoint_${STEP}" \
    --output "runs/official-sudoku-scaling-${SLURM_JOB_ID}-val-${STEP}.json" \
    --num-samples 1500 --batch-size 32 --steps 64 --cfgs 2
done

"$PY" -u official_sudoku_eval.py \
  --config official-configs/sudoku9_capability.yml \
  --checkpoint "$RUN/checkpoint_50000" \
  --data-path data/official-sudoku9-difficulty-v1/train \
  --output "runs/official-sudoku-scaling-${SLURM_JOB_ID}-train-50000.json" \
  --num-samples 1500 --batch-size 32 --steps 64 --cfgs 2
