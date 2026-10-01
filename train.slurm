#!/bin/bash
#SBATCH --job-name=flow-compose
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100_debug
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:a100_80gb_pcie:1
#SBATCH --chdir=/data/leuven/377/vsc37788/continuous-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=flow-compose-%j.out
set -euo pipefail

# Submit from the project directory. This is a bounded 2,000-step GPU pilot.
# Override interpreter using --export=ALL,FLOW_PYTHON=/absolute/path/to/python
cd /data/leuven/377/vsc37788/continuous-reasoning
export NETRC="${NETRC:-/data/leuven/377/vsc37788/.wandb-netrc}"
FLOW_PYTHON="${FLOW_PYTHON:-/data/leuven/377/vsc37788/miniconda3/envs/latentdiff-gpu/bin/python}"
"$FLOW_PYTHON" train.py --data data/pilot --out "runs/pilot-${SLURM_JOB_ID}" \
    --steps 2000 --batch-size 64 --eval-every 500 --sampling-steps 16 \
    --threads 4 --device cuda --self-condition --data-mode online \
    --wandb-mode "${FLOW_WANDB_MODE:-online}" \
    --wandb-project "${WANDB_PROJECT:-latent-diffusion-reasoning}" \
    --wandb-entity "${WANDB_ENTITY:-nlp_louise}"
