#!/bin/bash
#SBATCH --job-name=autonomous-correction
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --chdir=/data/leuven/377/vsc37788/continuous-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=56G
#SBATCH --time=01:00:00
#SBATCH --output=autonomous-correction-%A_%a.out
#SBATCH --array=0-2

set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/triton-${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-${SLURM_JOB_ID}
PY=/data/leuven/377/vsc37788/envs/elf-official/bin/python

EXTRA=()
if [[ "$SLURM_ARRAY_TASK_ID" == 0 ]]; then
  NAME=graph
  CONFIG=official-configs/graph_depth_capability.yml
  CHECKPOINT=runs/official-graph-capability-62169627/checkpoint_50000
  CFG=2
  SAMPLES=128
  BATCH=8
elif [[ "$SLURM_ARRAY_TASK_ID" == 1 ]]; then
  NAME=countdown
  CONFIG=official-configs/countdown_capability.yml
  CHECKPOINT=runs/official-countdown-capability-62169682/checkpoint_50000
  CFG=1
  SAMPLES=128
  BATCH=8
else
  NAME=sudoku-easy
  CONFIG=official-configs/sudoku9_capability.yml
  CHECKPOINT=runs/official-sudoku9-capability-62169683/checkpoint_100000
  CFG=1
  SAMPLES=96
  BATCH=1
  EXTRA=(--difficulty easy)
fi

OUT="runs/official-${NAME}-autonomous-correction-${SLURM_JOB_ID}"
"$PY" -u official_autonomous_correction.py \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" --output-dir "$OUT" \
  --num-samples "$SAMPLES" --batch-size "$BATCH" --restarts 4 \
  --flow-steps 16 --time-schedule uniform --cfg "$CFG" "${EXTRA[@]}"
