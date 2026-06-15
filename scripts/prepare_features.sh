#!/usr/bin/env bash
set -euo pipefail

python -m eeg_cogcappro.features \
  --data-dir "${DATA_DIR:-image-eeg-data}" \
  --image-root "${IMAGE_ROOT:-auto}" \
  --clip-backbone "${CLIP_BACKBONE:-RN50}" \
  --output-cache "${FEATURE_CACHE:-cache/features_rn50.pt}"
