#!/bin/bash
#SBATCH --job-name=variants
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/variants_%j.out
#SBATCH --error=logs/variants_%j.err

set -euo pipefail
mkdir -p logs

BASE="/hpc2hdd/JH_DATA/jhai_data/dsaa2012_031/project_codex"
cd "$BASE"
source /hpc2hdd/home/dsaa2012_031/miniconda3/etc/profile.d/conda.sh
conda activate eeg

bash scripts/train_variants_3seeds.sh
