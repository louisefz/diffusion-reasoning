#!/bin/bash
#SBATCH --job-name=lookup-elfscale
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --account=lp_zhou_diffusionlang
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --chdir=/data/leuven/377/vsc37788/continuous-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=lookup-elfscale-%j.out
set -euo pipefail
cd /data/leuven/377/vsc37788/continuous-reasoning
export NETRC=/data/leuven/377/vsc37788/.wandb-netrc
FLOW_OBJECTIVE="${FLOW_OBJECTIVE:?Set flow or direct}"
case "$FLOW_OBJECTIVE" in flow|direct) ;; *) exit 2 ;; esac
/data/leuven/377/vsc37788/miniconda3/envs/latentdiff-gpu/bin/python -u train.py \
    --data data/pilot --out "runs/lookup-elfscale-${FLOW_OBJECTIVE}-${SLURM_JOB_ID}" \
    --task lookup --objective "$FLOW_OBJECTIVE" \
    --steps 100000 --batch-size 64 --width 864 --layers 12 --heads 12 \
    --eval-every 1000 --checkpoint-every 25000 --sampling-steps 16 \
    --threads 8 --device cuda --self-condition --data-mode online \
    --lr 0.0003 --seed 0 --wandb-mode online \
    --wandb-project latent-diffusion-reasoning --wandb-entity nlp_louise \
    --wandb-group lookup-elfscale-100k
