#!/usr/bin/env bash
set -euo pipefail

python -m eeg_cogcappro.reconstruct \
  --data-dir "${DATA_DIR:-image-eeg-data}" \
  --feature-cache "${FEATURE_CACHE:-cache/features_rn50.pt}" \
  --ckpt "${CKPT:-runs/seed0/best.pt}" \
  --output-dir "${OUTPUT_DIR:-recons/seed0}" \
  --method "${METHOD:-auto}"
