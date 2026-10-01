#!/bin/bash
#SBATCH --job-name=composition-curve
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
#SBATCH --output=composition-curve-%j.out
set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
RUN_DIR=/data/leuven/377/vsc37788/continuous-reasoning/runs/official-composition-finetune-61958090
ARGS=()
for STEP in 3124 6248 9372 12496 15620 18744 21868 24992; do
  ARGS+=(--checkpoint "step-${STEP}=${RUN_DIR}/checkpoint_${STEP}")
done
/data/leuven/377/vsc37788/envs/elf-official/bin/python official_flow_sweep.py \
  --config official-configs/composition.yml \
  "${ARGS[@]}" \
  --num-samples 2400 \
  --schedules logit_normal \
  --steps 16 \
  --cfgs 1 \
  --output runs/official-composition-checkpoint-curve-${SLURM_JOB_ID}.json
