#!/usr/bin/env bash
set -euo pipefail

# Train and evaluate ATM-S experts for each new encoder backbone
# Usage: bash scripts/train_multi_experts.sh

DATA_DIR="${DATA_DIR:-image-eeg-data}"
DEVICE="${DEVICE:-cuda}"

echo "=== Phase 1: ATM-S expert training for new encoders (3 seeds for quick validation) ==="

# --- RN50 ---
FEATURE_CACHE_RN50="${FEATURE_CACHE_RN50:-cache/features_multi.pt}"
if [ ! -f "$FEATURE_CACHE_RN50" ]; then
    echo "WARNING: $FEATURE_CACHE_RN50 not found. Run prepare_multi_features.sh first."
    echo "Falling back to existing RN50 cache if available..."
    FEATURE_CACHE_RN50="cache/features_rn50.pt"
fi

for seed in 0 1 2; do
    echo "--- RN50 seed $seed ---"
    python -m eeg_cogcappro.train_atms \
        --config configs/atms_rn50.yaml \
        --data-dir "$DATA_DIR" \
        --feature-cache "$FEATURE_CACHE_RN50" \
        --feature-key rn50_feature \
        --seed "$seed" \
        --output-dir "runs/atms_rn50_seed${seed}" \
        --device "$DEVICE" \
        --save-last-as-best

    python -m eeg_cogcappro.eval_atms \
        --data-dir "$DATA_DIR" \
        --feature-cache "$FEATURE_CACHE_RN50" \
        --feature-key rn50_feature \
        --ckpt "runs/atms_rn50_seed${seed}/best.pt" \
        --split test \
        --output "results/atms_rn50_seed${seed}_test.json" \
        --device "$DEVICE"
done

# --- ViT-B/32 ---
FEATURE_CACHE_VITB32="${FEATURE_CACHE_VITB32:-cache/features_multi.pt}"
if [ ! -f "$FEATURE_CACHE_VITB32" ]; then
    echo "WARNING: $FEATURE_CACHE_VITB32 not found."
    FEATURE_CACHE_VITB32="cache/features_vitb32.pt"
fi

for seed in 0 1 2; do
    echo "--- ViT-B/32 seed $seed ---"
    python -m eeg_cogcappro.train_atms \
        --config configs/atms_vitb32.yaml \
        --data-dir "$DATA_DIR" \
        --feature-cache "$FEATURE_CACHE_VITB32" \
        --feature-key vit_b_32_feature \
        --seed "$seed" \
        --output-dir "runs/atms_vitb32_seed${seed}" \
        --device "$DEVICE" \
        --save-last-as-best

    python -m eeg_cogcappro.eval_atms \
        --data-dir "$DATA_DIR" \
        --feature-cache "$FEATURE_CACHE_VITB32" \
        --feature-key vit_b_32_feature \
        --ckpt "runs/atms_vitb32_seed${seed}/best.pt" \
        --split test \
        --output "results/atms_vitb32_seed${seed}_test.json" \
        --device "$DEVICE"
done

# --- DINOv2-da2 ---
FEATURE_CACHE_DINO="${FEATURE_CACHE_DINO:-cache/features_multi.pt}"
if [ ! -f "$FEATURE_CACHE_DINO" ]; then
    echo "WARNING: $FEATURE_CACHE_DINO not found."
    FEATURE_CACHE_DINO="cache/features_dinov2_da2.pt"
fi

for seed in 0 1 2; do
    echo "--- DINOv2-da2 seed $seed ---"
    python -m eeg_cogcappro.train_atms \
        --config configs/atms_dinov2.yaml \
        --data-dir "$DATA_DIR" \
        --feature-cache "$FEATURE_CACHE_DINO" \
        --feature-key dinov2_da2_feature \
        --seed "$seed" \
        --output-dir "runs/atms_dinov2_seed${seed}" \
        --device "$DEVICE" \
        --save-last-as-best

    python -m eeg_cogcappro.eval_atms \
        --data-dir "$DATA_DIR" \
        --feature-cache "$FEATURE_CACHE_DINO" \
        --feature-key dinov2_da2_feature \
        --ckpt "runs/atms_dinov2_seed${seed}/best.pt" \
        --split test \
        --output "results/atms_dinov2_seed${seed}_test.json" \
        --device "$DEVICE"
done

# --- VAE ---
FEATURE_CACHE_VAE="${FEATURE_CACHE_VAE:-cache/features_multi.pt}"
if [ ! -f "$FEATURE_CACHE_VAE" ]; then
    echo "WARNING: $FEATURE_CACHE_VAE not found."
    FEATURE_CACHE_VAE="cache/features_vae.pt"
fi

for seed in 0 1 2; do
    echo "--- VAE seed $seed ---"
    python -m eeg_cogcappro.train_atms \
        --config configs/atms_vae.yaml \
        --data-dir "$DATA_DIR" \
        --feature-cache "$FEATURE_CACHE_VAE" \
        --feature-key vae_feature \
        --seed "$seed" \
        --output-dir "runs/atms_vae_seed${seed}" \
        --device "$DEVICE" \
        --save-last-as-best

    python -m eeg_cogcappro.eval_atms \
        --data-dir "$DATA_DIR" \
        --feature-cache "$FEATURE_CACHE_VAE" \
        --feature-key vae_feature \
        --ckpt "runs/atms_vae_seed${seed}/best.pt" \
        --split test \
        --output "results/atms_vae_seed${seed}_test.json" \
        --device "$DEVICE"
done

echo "=== Phase 1 complete: all 3-seed experts trained and evaluated ==="