#!/bin/bash
#SBATCH --job-name=elf-shallow-heads
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:h100_80gb_hbm3:1
#SBATCH --chdir=/data/leuven/377/vsc37788/continuous-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=shallow-head-scan-%j.out

set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export XDG_CACHE_HOME=/tmp/cache-${SLURM_JOB_ID}
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
for depth in 1 2; do
  /data/leuven/377/vsc37788/envs/elf-official/bin/python -u official_head_patching.py \
    --config official-configs/composition_d4_curriculum_1m.yml \
    --checkpoint runs/official-composition-d4-1m-61996829/checkpoint_100000 \
    --output-dir "runs/official-shallow-head-d${depth}-${SLURM_JOB_ID}" \
    --depth "${depth}" --block 10 --pairs-per-step 100 --batch-size 20
done
