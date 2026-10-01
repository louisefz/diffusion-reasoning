#!/bin/bash
#SBATCH --job-name=d4-ddp-smoke
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100_debug
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:a100_80gb_pcie:1
#SBATCH --chdir=/data/leuven/377/vsc37788/continuous-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=d4-ddp-smoke-%j.out

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

set +e
timeout --signal=TERM 600s \
  /data/leuven/377/vsc37788/envs/elf-official/bin/python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=1 \
  official-elf/src/train.py \
  --config official-configs/composition_d4_curriculum_1m.yml \
  --config_override "output_dir=/tmp/official-composition-d4-smoke-${SLURM_JOB_ID}" \
  --config_override use_wandb=false \
  --config_override online_eval=false
status=$?
set -e

if [[ "$status" -eq 124 || "$status" -eq 143 ]]; then
  echo "SMOKE_TIMEOUT_SUCCESS: training remained alive for 600 seconds"
  exit 0
fi
exit "$status"
