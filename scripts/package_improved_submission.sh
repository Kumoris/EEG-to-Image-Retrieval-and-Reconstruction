#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"

"$PYTHON" scripts/package_submission.py \
  --output-dir outputs/atms_multimodal_final_improved \
  --retrieval-dir results/multi_encoder_ensemble \
  --recon-dir recons/atms_multimodal_final_improved \
  --zip-path outputs/atms_multimodal_final_improved/submission.zip \
  --split test \
  --expected-recons 200
