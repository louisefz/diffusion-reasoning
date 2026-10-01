#!/bin/bash
#SBATCH --job-name=elf-circuit-depth
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:h100_80gb_hbm3:1
#SBATCH --chdir=/data/leuven/377/vsc37788/continuous-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=circuit-depth-compare-%j.out

set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export XDG_CACHE_HOME=/tmp/cache-${SLURM_JOB_ID}
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
pairs=${PAIRS_PER_STEP:-50}
label=${RUN_LABEL:-full}

for spec in "1 10 1,2,3 4,5,6" "2 10 1,2,3 4,5,6" "4 11 3 4"; do
  read -r depth block circuit control <<< "${spec}"
  common=(
    --config official-configs/composition_d4_curriculum_1m.yml
    --checkpoint runs/official-composition-d4-1m-61996829/checkpoint_100000
    --depth "${depth}" --block "${block}"
    --circuit-heads "${circuit}" --control-heads "${control}"
    --pairs-per-step "${pairs}" --batch-size 20 --flow-steps 8
  )
  /data/leuven/377/vsc37788/envs/elf-official/bin/python -u official_circuit_response_kernel.py \
    "${common[@]}" --output-dir "runs/official-circuit-depth-d${depth}-kernel-${label}-${SLURM_JOB_ID}" \
    --strength 1.0
  /data/leuven/377/vsc37788/envs/elf-official/bin/python -u official_circuit_response_kernel.py \
    "${common[@]}" --output-dir "runs/official-circuit-depth-d${depth}-dose-${label}-${SLURM_JOB_ID}" \
    --time-index 0 --time-index 1 --time-index 2 \
    --strength 0.25 --strength 0.4 --strength 0.5 --strength 0.6 \
    --strength 0.75 --strength 0.9 --strength 1.0
done
