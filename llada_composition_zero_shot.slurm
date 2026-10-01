#!/bin/bash
#SBATCH --job-name=llada-comp-zs
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --chdir=/data/leuven/377/vsc37788/continuous-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=llada-composition-zs-%j.out

set -euo pipefail

cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/hf_cache
export HF_HUB_CACHE=/data/leuven/377/vsc37788/hf_cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME=/tmp/llada-cache-${SLURM_JOB_ID}
export TRITON_CACHE_DIR=/tmp/llada-triton-${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false

MODEL_DIR=$(find "${HF_HUB_CACHE}/models--GSAI-ML--LLaDA-8B-Instruct/snapshots" \
  -mindepth 1 -maxdepth 1 -type d | head -n 1)
if [[ -z "${MODEL_DIR}" ]]; then
  echo "LLaDA snapshot is missing" >&2
  exit 2
fi

/data/leuven/377/vsc37788/miniconda3/envs/dllm-cu128/bin/python -u \
  llada_eval_composition.py \
  --model "${MODEL_DIR}" \
  --output "runs/llada-composition-zero-shot-${SLURM_JOB_ID}-${OUTPUT_TAG:-full}/predictions.jsonl" \
  --tasks compose \
  --depths ${DEPTHS:-1 2 4 8} \
  --per-group "${PER_GROUP:-200}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --steps "${STEPS:-8}" \
  --gen-length "${GEN_LENGTH:-8}" \
  --block-length "${BLOCK_LENGTH:-8}"
