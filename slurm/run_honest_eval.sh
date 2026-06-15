#!/usr/bin/env bash
#SBATCH --job-name=honest_eval
#SBATCH --partition=i64m1tga800ue
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/honest_eval_%j.out
#SBATCH --error=logs/honest_eval_%j.err

set -euo pipefail
cd /hpc2hdd/JH_DATA/jhai_data/dsaa2012_031/project_codex
mkdir -p logs

echo "=== Job $SLURM_JOB_ID on $(hostname) === $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU yet"

bash scripts/run_honest_eval_pipeline.sh
