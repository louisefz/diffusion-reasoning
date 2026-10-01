#!/bin/bash
#SBATCH --job-name=composition-eval
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --chdir=/data/leuven/377/vsc37788/continuous-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=composition-eval-%j.out
set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
SOURCE_JOB_ID="${SOURCE_JOB_ID:?Set SOURCE_JOB_ID}"
ELF_INIT="${ELF_INIT:?Set ELF_INIT}"
CHECKPOINT=/data/leuven/377/vsc37788/continuous-reasoning/runs/official-composition-overfit-${ELF_INIT}-${SOURCE_JOB_ID}/checkpoint_10000
OUT=runs/official-composition-overfit-eval-${ELF_INIT}-${SLURM_JOB_ID}
mkdir -p "$OUT"
/data/leuven/377/vsc37788/envs/elf-official/bin/python official_diagnostics.py \
  --config official-configs/composition_overfit.yml \
  --checkpoint "${ELF_INIT}=${CHECKPOINT}" \
  --output "$OUT/decoder-denoiser.json"
/data/leuven/377/vsc37788/envs/elf-official/bin/python official_flow_sweep.py \
  --config official-configs/composition_overfit.yml \
  --checkpoint "${ELF_INIT}=${CHECKPOINT}" \
  --output "$OUT/flow-sweep.json"
