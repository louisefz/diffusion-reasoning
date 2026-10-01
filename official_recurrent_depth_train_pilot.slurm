#!/bin/bash
#SBATCH --job-name=elf-depth-recur
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:h100_80gb_hbm3:1
#SBATCH --chdir=/data/leuven/377/vsc37788/continuous-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=recurrent-depth-train-%j.out

set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export NETRC=/data/leuven/377/vsc37788/.wandb-netrc
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export XDG_CACHE_HOME=/tmp/cache-${SLURM_JOB_ID}
export TRITON_CACHE_DIR=/tmp/triton-${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-${SLURM_JOB_ID}
export WANDB_DATA_DIR=/tmp
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

run_dir=/data/leuven/377/vsc37788/continuous-reasoning/runs/official-recurrent-depth-pilot-${SLURM_JOB_ID}
/data/leuven/377/vsc37788/envs/elf-official/bin/python -u official-elf/src/train.py \
  --config official-configs/composition_recurrent_pilot.yml \
  --config_override "output_dir=${run_dir}" \
  --config_override "wandb_run_name=official-recurrent-depth-pilot-${SLURM_JOB_ID}" \
  --config_override "wandb_tag=official-elf,recurrent,depth-conditioned,pilot" \
  --config_override "reasoning_loop_randomize=false" \
  --config_override "reasoning_loop_depth_conditioned=true"
