#!/bin/bash
#SBATCH --partition=gpu_a100_debug
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:a100_80gb_pcie:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=flow-layer-map-%j.out

set -euo pipefail

cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export XDG_CACHE_HOME=/tmp/cache-${SLURM_JOB_ID}
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}

output_dir="runs/official-flow-layer-map-${SLURM_JOB_ID}"
/data/leuven/377/vsc37788/envs/elf-official/bin/python -u official_flow_layer_map.py \
  --config official-configs/composition_d4_curriculum_1m.yml \
  --checkpoint runs/official-composition-d4-1m-61996829/checkpoint_100000 \
  --trajectory runs/official-flow-trajectory-62060730/trajectory_tensors.pt \
  --probes runs/official-layer-probe-full-62073605/ridge_probes.joblib \
  --output-dir "${output_dir}" \
  --batch-size 30
