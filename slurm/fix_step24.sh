#!/bin/bash
#SBATCH -p debug
#SBATCH -o logs/fix_step24_%j.out
#SBATCH -e logs/fix_step24_%j.err
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH -D /hpc2hdd/JH_DATA/jhai_data/dsaa2012_031/project_codex

source /hpc2hdd/home/dsaa2012_031/miniconda3/etc/profile.d/conda.sh
conda activate eeg

PYTHON=python3
DATA_DIR="image-eeg-data"
FEATURE_CACHE="cache/features_multi.pt"
DEVICE="cuda"

echo "=== Fix Step 2: Ensemble evaluation ==="
echo "Started at $(date)"

mkdir -p results/multi_encoder_ensemble

$PYTHON scripts/ensemble_retrieval.py \
    --modality "image=results/atms_vitl_seed*_test_tta0.logits.pt" \
    --modality "depth=results/atms_depth_vitl_seed*_test_tta0.logits.pt" \
    --modality "edge=results/atms_edge_vitl_seed*_test_tta0.logits.pt" \
    --modality "rn50=results/atms_rn50_seed*_test.logits.pt" \
    --modality "vitb32=results/atms_vitb32_seed*_test.logits.pt" \
    --modality "dinov2=results/atms_dinov2_seed*_test.logits.pt" \
    --modality "vae=results/atms_vae_seed*_test.logits.pt" \
    --weights "image=0.35" \
    --weights "depth=0.15" \
    --weights "edge=0.15" \
    --weights "rn50=0.10" \
    --weights "vitb32=0.10" \
    --weights "dinov2=0.10" \
    --weights "vae=0.05" \
    --normalize row_zscore \
    --output-dir results/multi_encoder_ensemble/

echo "Step 2 complete at $(date)"

echo ""
echo "=== Fix Step 4: VAE reconstruction ==="

VAE_CKPT="runs/atms_vae_seed0/best.pt"

$PYTHON -m eeg_cogcappro.reconstruct_vae \
    --data-dir "$DATA_DIR" \
    --feature-cache "$FEATURE_CACHE" \
    --feature-key vae_feature \
    --vae-ckpt "$VAE_CKPT" \
    --projector-ckpt "runs/vae_projector_seed0/best.pt" \
    --output-dir "recons/vae_seed0" \
    --method vae_decode \
    --device "$DEVICE"

echo "Step 4 complete at $(date)"
echo "=== Done ==="