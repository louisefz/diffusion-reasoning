#!/bin/bash
#SBATCH --job-name=elf-layer-probe
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100_debug
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:a100_80gb_pcie:1
#SBATCH --chdir=/data/leuven/377/vsc37788/continuous-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=layer-probe-%j.out

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
export OMP_NUM_THREADS=6

mode=${LAYER_PROBE_MODE:-full}
if [[ "${mode}" == "smoke" ]]; then
  settings=(--train-per-group 64 --eval-per-group 32 --batch-size 32 --alpha 1)
else
  settings=(--train-per-group 2048 --eval-per-group 200 --batch-size 64 --alpha 0.1 --alpha 1 --alpha 10 --alpha 100)
fi

/data/leuven/377/vsc37788/envs/elf-official/bin/python -u official_layerwise_probe.py \
  --config official-configs/composition_d4_curriculum_1m.yml \
  --checkpoint runs/official-composition-d4-1m-61996829/checkpoint_100000 \
  --output-dir runs/official-layer-probe-${mode}-${SLURM_JOB_ID} \
  "${settings[@]}"
