#!/bin/bash
#SBATCH --job-name=make-elf-lookup
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --chdir=/data/leuven/377/vsc37788/continuous-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=make-elf-lookup-%j.out
set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export TRANSFORMERS_OFFLINE=1
/data/leuven/377/vsc37788/envs/elf-official/bin/python -u make_official_lookup.py \
  --out data/official-lookup-v1 \
  --tokenizer cache/models/t5-small \
  --train 100000 --validation 2000 --test 2000 --seed 20260908
