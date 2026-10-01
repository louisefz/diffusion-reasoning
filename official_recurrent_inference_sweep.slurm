#!/bin/bash
#SBATCH --job-name=elf-recur-sweep
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
#SBATCH --output=recurrent-inference-sweep-%j.out

set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export HF_HOME=/data/leuven/377/vsc37788/continuous-reasoning/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export XDG_CACHE_HOME=/tmp/cache-${SLURM_JOB_ID}
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
samples=${NUM_SAMPLES:-2400}
label=${RUN_LABEL:-full}
checkpoint=${CHECKPOINT:-runs/official-composition-d4-1m-61996829/checkpoint_100000}
checkpoint_name=${CHECKPOINT_NAME:-baseline}
memory_tokens=${MEMORY_TOKENS:-0}
read -r -a reasoning_loops <<< "${REASONING_LOOPS:-1 2 4 6}"
inner_time_args=()
if [[ "${NO_INNER_TIME:-0}" == "1" ]]; then
  inner_time_args+=(--no-reasoning-inner-time)
fi
coupling_args=()
if [[ "${DIRECT_COUPLING:-0}" == "1" ]]; then
  coupling_args+=(--reasoning-memory-direct-coupling)
fi
bottleneck_args=()
if [[ "${MEMORY_BOTTLENECK:-0}" == "1" ]]; then
  bottleneck_args+=(--reasoning-memory-bottleneck)
fi

/data/leuven/377/vsc37788/envs/elf-official/bin/python -u official_flow_sweep.py \
  --config official-configs/composition_d4_curriculum_1m.yml \
  --checkpoint "${checkpoint_name}=${checkpoint}" \
  --num-samples "${samples}" --batch-size 50 \
  --schedules logit_normal --steps 16 --cfgs 1 \
  --reasoning-loops "${reasoning_loops[@]}" --reasoning-loop-scale 1.0 \
  --reasoning-memory-tokens "${memory_tokens}" "${inner_time_args[@]}" "${coupling_args[@]}" "${bottleneck_args[@]}" \
  --output "runs/official-recurrent-inference-${label}-${SLURM_JOB_ID}.json"
