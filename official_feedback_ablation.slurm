#!/bin/bash
#SBATCH --job-name=feedback-ablation
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=feedback-ablation-%A_%a.out

set -euo pipefail
cd /vsc-hard-mounts/leuven-data/377/vsc37788/continuous-reasoning
PY=/data/leuven/377/vsc37788/envs/elf-official/bin/python

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    NAME=graph
    CONFIG=official-configs/graph_depth_capability.yml
    CHECKPOINT=runs/official-graph-capability-62169627/checkpoint_50000
    EXTRA="--cfg 2 --num-samples 128 --batch-size 8"
    ;;
  1)
    NAME=countdown
    CONFIG=official-configs/countdown_capability.yml
    CHECKPOINT=runs/official-countdown-capability-62169682/checkpoint_50000
    EXTRA="--cfg 1 --num-samples 128 --batch-size 8"
    ;;
  2)
    NAME=sudoku-easy
    CONFIG=official-configs/sudoku9_capability.yml
    CHECKPOINT=runs/official-sudoku9-capability-62169683/checkpoint_100000
    EXTRA="--cfg 1 --num-samples 96 --batch-size 1 --difficulty easy"
    ;;
esac

"${PY}" -u official_feedback_ablation.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "runs/official-${NAME}-feedback-ablation-${SLURM_JOB_ID}" \
  --restarts 4 --flow-steps 16 --time-schedule uniform ${EXTRA}
