#!/bin/bash
#SBATCH --job-name=elf-route-layer
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
#SBATCH --output=routing-layer-%j.out

set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME=/tmp/cache-${SLURM_JOB_ID}
export TRITON_CACHE_DIR=/tmp/triton-${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-${SLURM_JOB_ID}
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}
export OMP_NUM_THREADS=6

mode=${ROUTING_LAYER_MODE:-full}
if [[ "${mode}" == "smoke" ]]; then settings=(--pairs-per-step 20 --batch-size 20)
else settings=(--pairs-per-step 150 --batch-size 50); fi

/data/leuven/377/vsc37788/envs/elf-official/bin/python -u official_routing_layer_patching.py \
  --config official-configs/composition_d4_curriculum_1m.yml \
  --checkpoint runs/official-composition-d4-1m-61996829/checkpoint_100000 \
  --output-dir runs/official-routing-layer-${mode}-${SLURM_JOB_ID} \
  --max-layer 10 "${settings[@]}"
