#!/bin/bash
#SBATCH --job-name=sudoku-error-recovery
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --chdir=/data/leuven/377/vsc37788/continuous-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=sudoku-error-recovery-%j.out

set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export XDG_CACHE_HOME=/tmp/cache-${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false

NUM_SAMPLES=${NUM_SAMPLES:-64}
BATCH_SIZE=${BATCH_SIZE:-2}
RESTARTS=${RESTARTS:-2}
VARIANT=${VARIANT:-full}
EXTRA_ARGS=()
if [[ "${VARIANT}" == "pilot" ]]; then
  EXTRA_ARGS+=(--time-index 1 --time-index 4 --time-index 8)
  EXTRA_ARGS+=(--error-count 1 --error-count 3)
fi

/data/leuven/377/vsc37788/envs/elf-official/bin/python -u official_sudoku_error_injection.py \
  --config official-configs/sudoku9_capability.yml \
  --checkpoint runs/official-sudoku9-capability-62169683/checkpoint_200000 \
  --output-dir "runs/official-sudoku-error-recovery-${VARIANT}-${SLURM_JOB_ID}" \
  --difficulty hard --num-samples "${NUM_SAMPLES}" --batch-size "${BATCH_SIZE}" \
  --restarts "${RESTARTS}" --flow-steps 16 --time-schedule uniform --cfg 1 \
  --seed 20261001 "${EXTRA_ARGS[@]}"
