#!/bin/bash
#SBATCH --job-name=elf-memory-continue
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
#SBATCH --output=recurrent-memory-continue-%j.out

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

run_dir=/data/leuven/377/vsc37788/continuous-reasoning/runs/official-recurrent-memory-continue-${SLURM_JOB_ID}
/data/leuven/377/vsc37788/envs/elf-official/bin/python -u official-elf/src/train.py \
  --config official-configs/composition_recurrent_pilot.yml \
  --config_override "output_dir=${run_dir}" \
  --config_override "wandb_run_name=official-recurrent-memory-continue-${SLURM_JOB_ID}" \
  --config_override "wandb_tag=official-elf,recurrent,memory-bottleneck,continue" \
  --config_override "init_checkpoint=/data/leuven/377/vsc37788/continuous-reasoning/runs/official-recurrent-memory-pilot-62166469/checkpoint_2000" \
  --config_override "reasoning_loop_randomize=false" \
  --config_override "reasoning_loop_depth_conditioned=true" \
  --config_override "reasoning_aux_loss_weight=0.001" \
  --config_override "reasoning_memory_tokens=4" \
  --config_override "reasoning_memory_inner_time=true" \
  --config_override "reasoning_memory_direct_coupling=true" \
  --config_override "reasoning_memory_bottleneck=true" \
  --config_override "init_allow_missing=false" \
  --config_override "lr=0.00001" \
  --config_override "warmup_steps=100" \
  --config_override "save_freq=0.025" \
  --config_override "max_train_steps=5000"
