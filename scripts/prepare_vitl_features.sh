#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-python3}"

"$PYTHON" -m eeg_cogcappro.features \
  --data-dir "${DATA_DIR:-image-eeg-data}" \
  --image-root "${IMAGE_ROOT:-auto}" \
  --clip-backbone "${CLIP_BACKBONE:-ViT-L-14}" \
  --clip-pretrained "${CLIP_PRETRAINED:-laion2b_s32b_b82k}" \
  --output-cache "${FEATURE_CACHE:-cache/features_vitl.pt}" \
  --batch-size "${BATCH_SIZE:-256}" \
  --feature-dim "${FEATURE_DIM:-768}" \
  --device "${DEVICE:-cuda}" \
  --clean-only
