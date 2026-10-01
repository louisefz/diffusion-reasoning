#!/bin/bash
#SBATCH --job-name=flow-lookup
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
#SBATCH --output=flow-lookup-%j.out
set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export NETRC=/data/leuven/377/vsc37788/.wandb-netrc
/data/leuven/377/vsc37788/miniconda3/envs/latentdiff-gpu/bin/python train.py \
    --data data/pilot --out "runs/lookup-${SLURM_JOB_ID}" --task lookup \
    --objective "${FLOW_OBJECTIVE:-flow}" \
    --steps 2000 --batch-size 64 --width 128 --layers 4 \
    --eval-every 500 --sampling-steps 16 --threads 4 --device cuda \
    --self-condition --data-mode online --wandb-mode online \
    --wandb-project latent-diffusion-reasoning --wandb-entity nlp_louise \
    --wandb-group lookup-diagnostic
