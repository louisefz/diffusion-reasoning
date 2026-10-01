#!/bin/bash
#SBATCH --job-name=elf-pos-state
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100_debug
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:a100_80gb_pcie:1
#SBATCH --chdir=/data/leuven/377/vsc37788/continuous-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=position-state-%j.out

set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export XDG_CACHE_HOME=/tmp/cache-${SLURM_JOB_ID}
export TRITON_CACHE_DIR=/tmp/triton-${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-${SLURM_JOB_ID}
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

mode=${POSITION_PROBE_MODE:-full}
if [[ "${mode}" == "smoke" ]]; then
  settings=(--train-samples 256 --eval-samples 64 --batch-size 64 --alpha 1)
else
  # A single preregistered regularization value avoids selecting 520 probes
  # independently and keeps the full semantic map within the debug time limit.
  settings=(--train-samples 4096 --eval-samples 200 --batch-size 64 --alpha 1)
fi

/data/leuven/377/vsc37788/envs/elf-official/bin/python -u official_position_state_probe.py \
  --config official-configs/composition_d4_curriculum_1m.yml \
  --checkpoint runs/official-composition-d4-1m-61996829/checkpoint_100000 \
  --output-dir runs/official-position-state-${mode}-${SLURM_JOB_ID} \
  "${settings[@]}"
