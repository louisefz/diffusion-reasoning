#!/bin/bash
#SBATCH --job-name=graph-trajectory
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100_debug
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:a100_80gb_pcie:1
#SBATCH --chdir=/data/leuven/377/vsc37788/continuous-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --output=graph-trajectory-%j.out

set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning

export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/triton-${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-${SLURM_JOB_ID}

OUT="runs/official-graph-trajectory-${SLURM_JOB_ID}"
/data/leuven/377/vsc37788/envs/elf-official/bin/python -u official_flow_trajectory.py \
  --config official-configs/graph_depth_capability.yml \
  --checkpoint runs/official-graph-capability-62169627/checkpoint_50000 \
  --output-dir "$OUT" \
  --group lookup:4:100 \
  --group reach:1:100 \
  --group reach:4:100 \
  --group reach:6:100 \
  --group reach:8:100 \
  --group reach:12:100 \
  --batch-size 32 \
  --steps 16 \
  --schedule logit_normal \
  --cfg 2 \
  --no-save-tensors
