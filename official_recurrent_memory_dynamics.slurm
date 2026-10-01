#!/bin/bash
#SBATCH --job-name=elf-memory-dynamics
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:h100_80gb_hbm3:1
#SBATCH --chdir=/data/leuven/377/vsc37788/continuous-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=recurrent-memory-dynamics-%j.out

set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export XDG_CACHE_HOME=/tmp/cache-${SLURM_JOB_ID}
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

/data/leuven/377/vsc37788/envs/elf-official/bin/python -u official_recurrent_memory_dynamics.py \
  --config official-configs/composition_d4_curriculum_1m.yml \
  --checkpoint runs/official-recurrent-memory-continue-62166495/checkpoint_5000 \
  --output-dir runs/official-recurrent-memory-dynamics-${SLURM_JOB_ID} \
  --samples 200 --batch-size 50 --outer-steps 16 --schedule logit_normal \
  --inner-loops 4 --memory-tokens 4
