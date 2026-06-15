#!/usr/bin/env bash
set -euo pipefail

echo "=== Extracting VAE latent features ==="
python -m eeg_cogcappro.features multi \
  --data-dir "${DATA_DIR:-image-eeg-data}" \
  --image-root "${IMAGE_ROOT:-auto}" \
  --backends sd_vae \
  --vae-name "${VAE_NAME:-stabilityai/sd-vae-ft-mse}" \
  --output-cache "${OUTPUT_CACHE:-cache/features_vae.pt}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --feature-dim 512 \
  --device "${DEVICE:-cuda}"

echo "=== VAE feature cache written ==="