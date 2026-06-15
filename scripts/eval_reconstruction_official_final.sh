#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"

"$PYTHON" -m eeg_cogcappro.eval_reconstruction_official \
  --fake-dir recons/atms_multimodal_final \
  --output results/atms_multimodal_final_reconstruction_official.json \
  --metrics all \
  --batch-size 32 \
  --device auto \
  --allow-open-clip-fallback
