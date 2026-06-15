#!/usr/bin/env bash
set -euo pipefail

echo "=== Extracting RN50 CLIP features ==="
python -m eeg_cogcappro.features single \
  --data-dir "${DATA_DIR:-image-eeg-data}" \
  --image-root "${IMAGE_ROOT:-auto}" \
  --clip-backbone RN50 \
  --clip-pretrained openai \
  --output-cache "${OUTPUT_CACHE:-cache/features_rn50_standalone.pt}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --feature-dim 512 \
  --device "${DEVICE:-cuda}"

echo "=== RN50 feature cache written ==="