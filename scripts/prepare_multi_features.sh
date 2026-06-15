#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-image-eeg-data}"
IMAGE_ROOT="${IMAGE_ROOT:-auto}"
DEVICE="${DEVICE:-cuda}"
FEATURE_DIM="${FEATURE_DIM:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"
OUTPUT_CACHE="${OUTPUT_CACHE:-cache/features_multi.pt}"
PYTHON="${PYTHON:-python3}"

echo "=== Extracting multi-backend features (RN50 + ViT-B/32 + DINOv2_da2 + VAE) ==="

"$PYTHON" -m eeg_cogcappro.features multi \
  --data-dir "$DATA_DIR" \
  --image-root "$IMAGE_ROOT" \
  --backends RN50 ViT-B-32 dinov2_da2 sd_vae \
  --output-cache "$OUTPUT_CACHE" \
  --batch-size "$BATCH_SIZE" \
  --feature-dim "$FEATURE_DIM" \
  --device "$DEVICE"

echo "=== Multi-feature cache written to $OUTPUT_CACHE ==="
