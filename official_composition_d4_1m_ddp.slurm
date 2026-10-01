#!/bin/bash
#SBATCH --job-name=composition-d4-1m
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:a100-sxm4-80gb:2
#SBATCH --chdir=/data/leuven/377/vsc37788/continuous-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=composition-d4-1m-%j.out

set -euo pipefail

cd /data/leuven/377/vsc37788/continuous-reasoning
export NETRC=/data/leuven/377/vsc37788/.wandb-netrc
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME=/tmp/cache-${SLURM_JOB_ID}
export TRITON_CACHE_DIR=/tmp/triton-${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-${SLURM_JOB_ID}
export WANDB_DATA_DIR=/tmp
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

/data/leuven/377/vsc37788/envs/elf-official/bin/python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  official-elf/src/train.py \
  --config official-configs/composition_d4_curriculum_1m.yml \
  --config_override "output_dir=/data/leuven/377/vsc37788/continuous-reasoning/runs/official-composition-d4-1m-${SLURM_JOB_ID}" \
  --config_override "wandb_run_name=official-composition-d4-1m-${SLURM_JOB_ID}"
