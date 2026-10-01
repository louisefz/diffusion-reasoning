#!/bin/bash
#SBATCH --job-name=elf-depth-smoke
#SBATCH --account=EUHPC_D39_034
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=00:20:00
#SBATCH --output=/leonardo_work/EUHPC_D39_034/continuous-reasoning/logs/depth-smoke-%j.out

set -euo pipefail

PROJECT=/leonardo_work/EUHPC_D39_034/continuous-reasoning
ENV_DIR=/leonardo_work/EUHPC_D39_034/envs/elf-reasoning
JOB_CACHE=/leonardo_scratch/large/userexternal/fzhou000/elf-cache-${SLURM_JOB_ID}
RUN=${PROJECT}/runs/leonardo-sudoku4-depth-smoke-${SLURM_JOB_ID}

mkdir -p "$JOB_CACHE" "$RUN"
cd "$PROJECT"

export HF_HOME=${PROJECT}/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export XDG_CACHE_HOME=${JOB_CACHE}/xdg
export TRITON_CACHE_DIR=${JOB_CACHE}/triton
export TORCHINDUCTOR_CACHE_DIR=${JOB_CACHE}/torchinductor
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

srun --ntasks=1 "$ENV_DIR/bin/python" -m torch.distributed.run \
  --standalone --nproc_per_node=4 official-elf/src/train.py \
  --config official-configs/leonardo_sudoku4_depth_extrapolation.yml \
  --config_override "output_dir=${RUN}" \
  --config_override "max_train_steps=10" \
  --config_override "use_wandb=false" \
  --config_override "log_freq=1" \
  --config_override "wandb_run_name=leonardo-sudoku4-depth-smoke-${SLURM_JOB_ID}"

echo "SMOKE_TEST_PASSED run=${RUN}"

