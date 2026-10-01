#!/bin/bash
#SBATCH --job-name=composition-branch
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
#SBATCH --output=composition-branch-%j.out
set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/triton-${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-${SLURM_JOB_ID}
/data/leuven/377/vsc37788/envs/elf-official/bin/python -u official_branching_diagnostic.py \
  --config official-configs/composition.yml \
  --checkpoint runs/official-composition-finetune-61958090/checkpoint_49984 \
  --groups lookup/d1 compose/d1 compose/d2 \
  --samples-per-group 20 --batch-size 12 --steps 16 \
  --branch-states 1 8 16 --sigmas 0 0.5 1 2 --replicas 2 \
  --output runs/official-composition-branch-${SLURM_JOB_ID}/branching.json
