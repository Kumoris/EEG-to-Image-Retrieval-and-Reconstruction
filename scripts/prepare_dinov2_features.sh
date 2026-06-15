#!/usr/bin/env bash
set -euo pipefail

echo "=== Extracting DINOv2-da2 features ==="
python -m eeg_cogcappro.features multi \
  --data-dir "${DATA_DIR:-image-eeg-data}" \
  --image-root "${IMAGE_ROOT:-auto}" \
  --backends dinov2_da2 \
  --dinov2-model "${DINOV2_MODEL:-dinov2_vitb14}" \
  --output-cache "${OUTPUT_CACHE:-cache/features_dinov2_da2.pt}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --feature-dim 512 \
  --device "${DEVICE:-cuda}"

echo "=== DINOv2-da2 feature cache written ==="