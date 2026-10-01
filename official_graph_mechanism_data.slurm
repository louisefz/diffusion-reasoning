#!/bin/bash
#SBATCH --job-name=graph-mech-data
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --chdir=/data/leuven/377/vsc37788/continuous-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=graph-mech-data-%j.out

set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

/data/leuven/377/vsc37788/envs/elf-official/bin/python -u \
  make_official_graph_mechanism.py \
  --reference data/official-graph-depth-v1 \
  --out data/official-graph-mechanism-v1 \
  --tokenizer cache/models/t5-small \
  --depths 4,8,12 \
  --episodes 3000
