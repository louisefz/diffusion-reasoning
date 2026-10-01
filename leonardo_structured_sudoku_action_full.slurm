#!/bin/bash
#SBATCH --job-name=structured-action-full
#SBATCH --account=EUHPC_D39_034
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/leonardo_work/EUHPC_D39_034/continuous-reasoning/logs/structured-action-full-%j.out

set -euo pipefail

PROJECT=/leonardo_work/EUHPC_D39_034/continuous-reasoning
ENV_DIR=/leonardo_work/EUHPC_D39_034/envs/elf-reasoning
JOB_CACHE=/leonardo_scratch/large/userexternal/fzhou000/structured-action-${SLURM_JOB_ID}
OUTPUT=${PROJECT}/results/structured-action-full-${SLURM_JOB_ID}

mkdir -p "$JOB_CACHE" "$OUTPUT" "${PROJECT}/logs"
cd "$PROJECT"
export HF_HOME=${PROJECT}/cache/huggingface
export HF_DATASETS_OFFLINE=1
export XDG_CACHE_HOME=${JOB_CACHE}/xdg
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

srun --ntasks=1 "$ENV_DIR/bin/python" -u structured_sudoku_action_flow.py \
  --train-data data/structured-sudoku9-action/train.npz \
  --test-data data/structured-sudoku9-action/test.npz \
  --output-dir "$OUTPUT" \
  --train-limit 200000 \
  --test-limit 10000 \
  --steps 20000 \
  --batch-size 256 \
  --eval-batch-size 512 \
  --evaluation-limit 4096 \
  --verifier-limit 1024 \
  --eval-every 2000 \
  --nfe 16

echo "STRUCTURED_ACTION_COMPLETED output=${OUTPUT}"
