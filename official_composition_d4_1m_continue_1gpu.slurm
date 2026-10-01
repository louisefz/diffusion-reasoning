#!/bin/bash
#SBATCH --job-name=d4-1m-continue
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --chdir=/data/leuven/377/vsc37788/continuous-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=composition-d4-1m-continue-%j.out

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

RUN_DIR=/data/leuven/377/vsc37788/continuous-reasoning/runs/official-composition-d4-1m-61996829

/data/leuven/377/vsc37788/envs/elf-official/bin/python -u official-elf/src/train.py \
  --config official-configs/composition_d4_curriculum_1m.yml \
  --config_override "output_dir=${RUN_DIR}" \
  --config_override "resume=${RUN_DIR}" \
  --config_override resume_reset_optimizer=true \
  --config_override init_checkpoint=none \
  --config_override wandb_run_name=official-composition-d4-1m-61996829 \
  --config_override wandb_resume=allow
