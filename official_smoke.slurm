#!/bin/bash
#SBATCH --job-name=official-elf-smoke
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100_debug
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:a100_80gb_pcie:1
#SBATCH --chdir=/data/leuven/377/vsc37788/continuous-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=00:30:00
#SBATCH --output=official-elf-smoke-%j.out
set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
/data/leuven/377/vsc37788/envs/elf-official/bin/python -u official-elf/src/eval.py \
  --config official-configs/owt_smoke.yml \
  --checkpoint_path cache/models/ELF-B-owt-torch/checkpoint_95085
