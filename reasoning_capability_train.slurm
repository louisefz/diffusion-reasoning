#!/bin/bash
#SBATCH --job-name=elf-capability
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --chdir=/data/leuven/377/vsc37788/continuous-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=capability-%A_%a.out
#SBATCH --array=0-2

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

case "${SLURM_ARRAY_TASK_ID}" in
  0) NAME=graph; CONFIG=official-configs/graph_depth_capability.yml ;;
  1) NAME=countdown; CONFIG=official-configs/countdown_capability.yml ;;
  2) NAME=sudoku9; CONFIG=official-configs/sudoku9_capability.yml ;;
esac
RUN=/data/leuven/377/vsc37788/continuous-reasoning/runs/official-${NAME}-capability-${SLURM_JOB_ID}

/data/leuven/377/vsc37788/envs/elf-official/bin/python -u official-elf/src/train.py \
  --config "${CONFIG}" \
  --config_override "output_dir=${RUN}" \
  --config_override "wandb_run_name=official-${NAME}-capability-${SLURM_JOB_ID}"
