#!/usr/bin/env bash
#SBATCH --job-name=eval_train
#SBATCH --partition=i64m1tga800ue
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_train_%j.out
#SBATCH --error=logs/eval_train_%j.err

set -euo pipefail
cd /hpc2hdd/JH_DATA/jhai_data/dsaa2012_031/project_codex

mkdir -p logs

echo "=== Job $SLURM_JOB_ID on $(hostname) === $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

bash scripts/eval_all_train.sh

echo "=== Done === $(date)"
