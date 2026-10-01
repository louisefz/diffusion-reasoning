#!/bin/bash
#SBATCH --job-name=composition-patch-hires
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100_debug
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:a100_80gb_pcie:1
#SBATCH --chdir=/data/leuven/377/vsc37788/continuous-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=composition-patch-hires-%j.out
set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/triton-${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-${SLURM_JOB_ID}
for SEED in 20260920 20260921 20260922; do
  /data/leuven/377/vsc37788/envs/elf-official/bin/python -u official_patching_diagnostic.py \
    --config official-configs/composition.yml \
    --checkpoint runs/official-composition-finetune-61958090/checkpoint_49984 \
    --groups lookup/d1 compose/d1 compose/d2 \
    --samples-per-group 200 --batch-size 20 --steps 16 \
    --branch-states 6 7 8 9 10 11 12 13 14 \
    --alphas 1.0 --scopes target_suffix \
    --modes z_only cache_only joint z_reset_cache \
    --seed "${SEED}" \
    --output "runs/official-composition-patch-hires-${SLURM_JOB_ID}-${SEED}/patching.json"
done
