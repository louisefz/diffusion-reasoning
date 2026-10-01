#!/bin/bash
#SBATCH --job-name=elf-branch-stability
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
#SBATCH --output=branch-stability-%j.out

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

mode=${BRANCH_MODE:-full}
if [[ "${mode}" == "smoke" ]]; then
  groups=(--group compose:1:2 --group compose:2:2 --group compose:4:2 --group lookup:4:2)
  settings=(--batch-size 2 --branches 2 --branch-index 2 --branch-index 14 --sigma 0 --sigma 0.03 --sigma 0.1 --perturb-target z --perturb-target selfcond --perturb-target both)
else
  groups=(--group compose:1:30 --group compose:2:30 --group compose:4:30 --group lookup:4:30)
  settings=(--batch-size 4 --branches 8 --branch-index 2 --branch-index 7 --branch-index 13 --branch-index 14 --sigma 0 --sigma 0.03 --sigma 0.1 --sigma 0.3 --sigma 0.5 --sigma 1.0 --perturb-target z --perturb-target selfcond --perturb-target both)
fi

/data/leuven/377/vsc37788/envs/elf-official/bin/python -u official_branching_stability.py \
  --config official-configs/composition_d4_curriculum_1m.yml \
  --checkpoint runs/official-composition-d4-1m-61996829/checkpoint_100000 \
  --output-dir runs/official-branch-stability-${mode}-${SLURM_JOB_ID} \
  --steps 16 \
  --schedule logit_normal \
  --cfg 2 \
  --self-cond-cfg 1 \
  "${groups[@]}" \
  "${settings[@]}"
