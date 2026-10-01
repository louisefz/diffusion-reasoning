#!/bin/bash
#SBATCH --job-name=sudoku4-depth-train
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:a100-sxm4-80gb:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=sudoku4-depth-train-%j.out

set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export NETRC=/data/leuven/377/vsc37788/.wandb-netrc
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export XDG_CACHE_HOME=/tmp/cache-${SLURM_JOB_ID}
export TRITON_CACHE_DIR=/tmp/triton-${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-${SLURM_JOB_ID}
export WANDB_DATA_DIR=/tmp/wandb-${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

RUN=/data/leuven/377/vsc37788/continuous-reasoning/runs/official-sudoku4-depth-extrapolation-${SLURM_JOB_ID}
mkdir -p "$WANDB_DATA_DIR"
/data/leuven/377/vsc37788/envs/elf-official/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 official-elf/src/train.py \
  --config official-configs/sudoku4_transition_short.yml \
  --config_override "output_dir=${RUN}" \
  --config_override "wandb_run_name=official-sudoku4-depth-extrapolation-${SLURM_JOB_ID}"
