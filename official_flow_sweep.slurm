#!/bin/bash
#SBATCH --job-name=official-flow-sweep
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100_debug
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:a100_80gb_pcie:1
#SBATCH --chdir=/data/leuven/377/vsc37788/continuous-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=official-flow-sweep-%j.out
set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/triton-${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-${SLURM_JOB_ID}
/data/leuven/377/vsc37788/envs/elf-official/bin/python -u official_flow_sweep.py \
  --config official-configs/lookup_pilot.yml \
  --checkpoint finetune=/data/leuven/377/vsc37788/continuous-reasoning/runs/official-lookup-finetune-61934533/checkpoint_1562 \
  --checkpoint scratch=/data/leuven/377/vsc37788/continuous-reasoning/runs/official-lookup-scratch-61934549/checkpoint_1562 \
  --num-samples 256 \
  --batch-size 64 \
  --output runs/official-flow-sweep-${SLURM_JOB_ID}/metrics.json
