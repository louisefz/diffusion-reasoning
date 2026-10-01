#!/bin/bash
#SBATCH --job-name=composition-heldout
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --chdir=/data/leuven/377/vsc37788/continuous-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=composition-heldout-%j.out
set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
CHECKPOINT_PATH="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH}"
MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME}"
OUT=runs/official-composition-heldout-${MODEL_NAME}-${SLURM_JOB_ID}.json
/data/leuven/377/vsc37788/envs/elf-official/bin/python official_flow_sweep.py \
  --config official-configs/composition.yml \
  --checkpoint "${MODEL_NAME}=${CHECKPOINT_PATH}" \
  --num-samples 2400 \
  --schedules logit_normal \
  --steps 16 \
  --cfgs 1 \
  --output "$OUT"
