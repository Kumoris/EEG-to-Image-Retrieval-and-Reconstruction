#!/usr/bin/env bash
set -euo pipefail

# Evaluate multi-encoder ensemble combining existing ViT-L multimodal + new encoder experts
# Uses existing ensemble_retrieval.py infrastructure

DEVICE="${DEVICE:-cuda}"
DATA_DIR="${DATA_DIR:-image-eeg-data}"
FEATURE_CACHE="${FEATURE_CACHE:-cache/features_multi.pt}"
PYTHON="${PYTHON:-python3}"

echo "=== Multi-encoder ensemble evaluation ==="

# Step 1: Evaluate existing ViT-L experts (will use existing logits if available)
# These should already exist from previous training runs
echo "Checking for existing ViT-L logits..."
VITL_LOGITS=""
for mod in image depth edge; do
    FOUND=0
    for seed in 0 1 2 3 4 5 6 7 8 9; do
        f="results/atms_${mod}_vitl_seed${seed}_test_tta0.logits.pt"
        if [ -f "$f" ]; then
            FOUND=1
            break
        fi
    done
    if [ "$FOUND" -eq "0" ]; then
        echo "WARNING: No ${mod} ViT-L logits found. Run existing ViT-L training first."
    fi
done

# Step 2: Evaluate new encoder experts when checkpoints are available.
echo "Evaluating new encoder experts on test set..."

for ENCODER in rn50 vitb32 dinov2 vae; do
    for seed in 0 1 2 3 4 5 6 7 8 9; do
        CKPT="runs/atms_${ENCODER}_seed${seed}/best.pt"
        if [ -f "$CKPT" ]; then
            KEY="${ENCODER}_feature"
            case "$ENCODER" in
                rn50) KEY="rn50_feature" ;;
                vitb32) KEY="vit_b_32_feature" ;;
                dinov2) KEY="dinov2_da2_feature" ;;
                vae) KEY="vae_feature" ;;
            esac
            "$PYTHON" -m eeg_cogcappro.eval_atms \
                --data-dir "$DATA_DIR" \
                --feature-cache "$FEATURE_CACHE" \
                --feature-key "$KEY" \
                --ckpt "$CKPT" \
                --split test \
                --output "results/atms_${ENCODER}_seed${seed}_test.json" \
                --device "$DEVICE"
        else
            echo "WARNING: $CKPT not found, skipping"
        fi
    done
done

# Step 3: Build multi-encoder ensemble
echo "Building multi-encoder ensemble..."

"$PYTHON" scripts/ensemble_retrieval.py \
  --modality "image=results/atms_vitl_seed*_test_tta0.logits.pt" \
  --modality "depth=results/atms_depth_vitl_seed*_test_tta0.logits.pt" \
  --modality "edge=results/atms_edge_vitl_seed*_test_tta0.logits.pt" \
  --modality "rn50=results/atms_rn50_seed*_test.logits.pt" \
  --modality "vitb32=results/atms_vitb32_seed*_test.logits.pt" \
  --modality "dinov2=results/atms_dinov2_seed*_test.logits.pt" \
  --modality "vae=results/atms_vae_seed*_test.logits.pt" \
  --weights "image=0.35" "depth=0.15" "edge=0.15" "rn50=0.10" "vitb32=0.10" "dinov2=0.10" "vae=0.05" \
  --normalize row_zscore \
  --output-dir results/multi_encoder_ensemble/

echo "=== Multi-encoder ensemble evaluation complete ==="
echo "Results in results/multi_encoder_ensemble/"
