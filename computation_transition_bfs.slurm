#!/bin/bash
#SBATCH --job-name=bfs-compute-transition
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=bfs-compute-transition-%A_%a.out

set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning

MODELS=(flow recurrent)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
RUN_DIR="runs/bfs-computation-transition-${MODEL}-${SLURM_ARRAY_JOB_ID}"

export WANDB_DIR=/data/leuven/377/vsc37788/continuous-reasoning/wandb
export WANDB_CACHE_DIR=/data/leuven/377/vsc37788/continuous-reasoning/cache/wandb
export WANDB_CONFIG_DIR=/data/leuven/377/vsc37788/continuous-reasoning/cache/wandb-config
export WANDB_DATA_DIR=/data/leuven/377/vsc37788/continuous-reasoning/cache/wandb-data
# Compute nodes do not inherit the interactive-shell API key. Keep a complete
# local W&B record; it can be synced after training without blocking the job.
export WANDB_MODE=offline
export XDG_CACHE_HOME=/tmp/cache-${SLURM_JOB_ID}

/vsc-hard-mounts/leuven-data/377/vsc37788/miniconda3/envs/latentdiff-gpu/bin/python -u \
  computation_transition_bfs.py \
  --model "$MODEL" \
  --output-dir "$RUN_DIR" \
  --steps 20000 \
  --batch-size 128 \
  --width 128 \
  --num-nodes 40 \
  --train-depth 4 \
  --eval-batches 16 \
  --eval-batch-size 64 \
  --wandb
