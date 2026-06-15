#!/bin/bash
#SBATCH -p i64m1tga40u
#SBATCH -o logs/step0_features_%j.out
#SBATCH -e logs/step0_features_%j.err
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH -D /hpc2hdd/JH_DATA/jhai_data/dsaa2012_031/project_codex

echo "=== Step 0: Extract multi-backend features ==="
echo "Job started at $(date)"
echo "Node: $(hostname)"

export PYTHONNOUSERSITE=0
export PATH="$HOME/.local/bin:$PATH"

mkdir -p cache logs

python3 -m eeg_cogcappro.features multi \
    --data-dir image-eeg-data \
    --backends RN50 ViT-B-32 dinov2_da2 sd_vae \
    --output-cache cache/features_multi.pt \
    --batch-size 32 \
    --feature-dim 512 \
    --device cuda

echo "=== Step 0 complete at $(date) ==="
echo "Feature cache saved to: cache/features_multi.pt"