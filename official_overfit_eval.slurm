#!/bin/bash
#SBATCH --job-name=official-overfit-eval
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --chdir=/data/leuven/377/vsc37788/continuous-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=official-overfit-eval-%j.out
set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/triton-${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-${SLURM_JOB_ID}
SOURCE_JOB_ID="${SOURCE_JOB_ID:?Set SOURCE_JOB_ID to the completed overfit job}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-5000}"
CHECKPOINT=/data/leuven/377/vsc37788/continuous-reasoning/runs/official-overfit-${SOURCE_JOB_ID}/checkpoint_${CHECKPOINT_STEP}
/data/leuven/377/vsc37788/envs/elf-official/bin/python -u official_diagnostics.py \
  --config official-configs/lookup_overfit.yml \
  --checkpoint overfit=${CHECKPOINT} \
  --num-samples 256 \
  --batch-size 64 \
  --output runs/official-overfit-eval-${SLURM_JOB_ID}/decoder-denoiser.json
/data/leuven/377/vsc37788/envs/elf-official/bin/python -u official_flow_sweep.py \
  --config official-configs/lookup_overfit.yml \
  --checkpoint overfit=${CHECKPOINT} \
  --num-samples 256 \
  --batch-size 64 \
  --output runs/official-overfit-eval-${SLURM_JOB_ID}/flow-sweep.json
