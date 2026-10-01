#!/bin/bash
#SBATCH --job-name=multipath-transition
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --output=multipath-transition-%A_%a.out

set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning

MODELS=(flow recurrent)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
RUN_DIR="runs/multipath-computation-transition-${MODEL}-${SLURM_ARRAY_JOB_ID}"

export WANDB_DIR=/data/leuven/377/vsc37788/continuous-reasoning/wandb
export WANDB_CACHE_DIR=/data/leuven/377/vsc37788/continuous-reasoning/cache/wandb
export WANDB_CONFIG_DIR=/data/leuven/377/vsc37788/continuous-reasoning/cache/wandb-config
export WANDB_DATA_DIR=/data/leuven/377/vsc37788/continuous-reasoning/cache/wandb-data
export WANDB_MODE=offline
export XDG_CACHE_HOME=/tmp/cache-${SLURM_JOB_ID}

/vsc-hard-mounts/leuven-data/377/vsc37788/miniconda3/envs/latentdiff-gpu/bin/python -u \
  computation_transition_multipath.py \
  --model "$MODEL" \
  --output-dir "$RUN_DIR" \
  --steps 20000 \
  --batch-size 128 \
  --width 128 \
  --route-width 3 \
  --max-nodes 96 \
  --train-depth 4 \
  --eval-graphs 64 \
  --eval-restarts 32 \
  --wandb
