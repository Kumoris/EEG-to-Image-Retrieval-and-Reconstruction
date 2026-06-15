#!/usr/bin/env bash
# Evaluate all trained models on train split to generate train logits
# for honest weight optimization (train-optimize, test-evaluate).
set -euo pipefail

BASE="/hpc2hdd/JH_DATA/jhai_data/dsaa2012_031/project_codex"
cd "$BASE"
source /hpc2hdd/home/dsaa2012_031/miniconda3/etc/profile.d/conda.sh
conda activate eeg

DATA_DIR="image-eeg-data"
CACHE_VITL="cache/features_vitl_real.pt"
CACHE_MULTI="cache/features_multi.pt"
TTA=5

eval_one() {
    local CKPT="$1" CACHE="$2" FKEY="$3" OUT_JSON="$4"
    local OUT_LOGITS="${OUT_JSON%.json}.logits.pt"
    if [ -f "$OUT_LOGITS" ]; then
        echo "SKIP $OUT_JSON (logits exist)"
        return
    fi
    echo "=== Eval train: $(basename "$CKPT") === $(date)"
    python3 -m eeg_cogcappro.eval_atms \
        --data-dir "$DATA_DIR" \
        --feature-cache "$CACHE" \
        --feature-key "$FKEY" \
        --ckpt "$CKPT" \
        --split train \
        --tta-n "$TTA" \
        --output "$OUT_JSON" \
        --device cuda
}

echo "=== Phase 1: Deep ViT-L image/depth/edge (10 seeds each) ==="
for CKEY in image_clean_feature depth_feature edge_feature; do
    case "$CKEY" in
        image_clean_feature) SHORT="image" ;;
        depth_feature) SHORT="depth" ;;
        edge_feature) SHORT="edge" ;;
    esac
    for SEED in $(seq 0 9); do
        eval_one \
            "runs/deep_vitl_${SHORT}_seed${SEED}/best.pt" \
            "$CACHE_VITL" \
            "$CKEY" \
            "results/deep_vitl_${SHORT}_seed${SEED}_train_tta5.json"
    done
done

echo "=== Phase 2: Deep RN50 / ViT-B32 / DINOv2 / VAE (3 seeds each) ==="
for SEED in $(seq 0 2); do
    eval_one "runs/deep_rn50_seed${SEED}/best.pt" "$CACHE_MULTI" "rn50_feature" "results/deep_rn50_seed${SEED}_train_tta5.json"
    eval_one "runs/deep_vit_b_32_seed${SEED}/best.pt" "$CACHE_MULTI" "vit_b_32_feature" "results/deep_vit_b_32_seed${SEED}_train_tta5.json"
    eval_one "runs/deep_dinov2_da2_seed${SEED}/best.pt" "$CACHE_MULTI" "dinov2_da2_feature" "results/deep_dinov2_da2_seed${SEED}_train_tta5.json"
    eval_one "runs/deep_vae_seed${SEED}/best.pt" "$CACHE_MULTI" "vae_feature" "results/deep_vae_seed${SEED}_train_tta5.json"
done

echo "=== All train logits generated === $(date)"
