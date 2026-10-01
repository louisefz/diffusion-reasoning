#!/bin/bash
#SBATCH --job-name=graph-sampling-sweep
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --chdir=/data/leuven/377/vsc37788/continuous-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=graph-sampling-sweep-%j.out

set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning

export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

CHECKPOINT_PATH=/data/leuven/377/vsc37788/continuous-reasoning/runs/official-graph-capability-62169627/checkpoint_50000
OUT="runs/official-graph-sampling-sweep-${SLURM_JOB_ID}.json"

/data/leuven/377/vsc37788/envs/elf-official/bin/python official_flow_sweep.py \
  --config official-configs/graph_depth_capability.yml \
  --checkpoint "graph-50k=${CHECKPOINT_PATH}" \
  --num-samples 3600 \
  --batch-size 64 \
  --schedules logit_normal \
  --steps 16 32 64 \
  --cfgs 1 2 \
  --output "$OUT"
