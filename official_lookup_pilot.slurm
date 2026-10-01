#!/bin/bash
#SBATCH --job-name=official-lookup
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --chdir=/data/leuven/377/vsc37788/continuous-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=official-lookup-%j.out
set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export NETRC=/data/leuven/377/vsc37788/.wandb-netrc
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/triton-${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-${SLURM_JOB_ID}
export WANDB_DATA_DIR=/tmp/wandb-data-${SLURM_JOB_ID}
ELF_INIT="${ELF_INIT:?Set scratch or finetune}"
case "$ELF_INIT" in
  scratch) INIT_PATH=none ;;
  finetune) INIT_PATH=/data/leuven/377/vsc37788/continuous-reasoning/cache/models/ELF-B-xsum-torch/checkpoint_39800 ;;
  *) exit 2 ;;
esac
/data/leuven/377/vsc37788/envs/elf-official/bin/python -u official-elf/src/train.py \
  --config official-configs/lookup_pilot.yml \
  --config_override "output_dir=/data/leuven/377/vsc37788/continuous-reasoning/runs/official-lookup-${ELF_INIT}-${SLURM_JOB_ID}" \
  --config_override "wandb_run_name=official-lookup-${ELF_INIT}-${SLURM_JOB_ID}" \
  --config_override "init_checkpoint=${INIT_PATH}"
