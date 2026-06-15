#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"

"$PYTHON" scripts/package_submission.py \
  --output-dir outputs/atms_multimodal_final \
  --retrieval-dir results/multi_encoder_ensemble \
  --recon-dir recons/atms_multimodal_final \
  --zip-path outputs/atms_multimodal_final/submission.zip \
  --split test \
  --expected-recons 200
