#!/bin/bash
#SBATCH --job-name=deep_vitl
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/deep_vitl_%j.out
#SBATCH --error=logs/deep_vitl_%j.err

set -euo pipefail
mkdir -p logs

BASE="/hpc2hdd/JH_DATA/jhai_data/dsaa2012_031/project_codex"
cd "$BASE"
source /hpc2hdd/home/dsaa2012_031/miniconda3/etc/profile.d/conda.sh
conda activate eeg

echo "=== Step 1: Extract real ViT-L features ===" "$(date)"
python3 -m eeg_cogcappro.features \
    --data-dir image-eeg-data \
    --image-root auto \
    --clip-backbone ViT-L-14 \
    --clip-pretrained laion2b_s32b_b82k \
    --output-cache cache/features_vitl_real.pt \
    --batch-size 256 \
    --feature-dim 768 \
    --device cuda

echo "=== Step 2: Train + eval Deep ATM-S ===" "$(date)"
bash scripts/train_deep_vitl_all.sh

echo "=== Step 3: Re-run ensemble with optimized weights ===" "$(date)"
python3 scripts/ensemble_retrieval.py \
    --normalize row_zscore \
    --modality "image=results/deep_vitl_image_seed*_test_tta5.logits.pt" \
    --modality "depth=results/deep_vitl_depth_seed*_test_tta5.logits.pt" \
    --modality "edge=results/deep_vitl_edge_seed*_test_tta5.logits.pt" \
    --modality "deep_rn50=results/deep_rn50_seed*_test_tta5.logits.pt" \
    --modality "deep_vitb32=results/deep_vit_b_32_seed*_test_tta5.logits.pt" \
    --modality "deep_dinov2=results/deep_dinov2_da2_seed*_test_tta5.logits.pt" \
    --modality "deep_vae=results/deep_vae_seed*_test_tta5.logits.pt" \
    --weights image=0.24 \
    --weights depth=0.15 \
    --weights edge=0.07 \
    --weights deep_rn50=0.11 \
    --weights deep_vitb32=0.07 \
    --weights deep_dinov2=0.14 \
    --weights deep_vae=0.21 \
    --output-dir results/deep_vitl_ensemble_v1

echo "=== Step 4: Grid search with new Deep ViT-L logits ===" "$(date)"
python3 scripts/grid_search_ensemble.py --results-dir results

echo "=== ALL DONE ===" "$(date)"
