#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"

"$PYTHON" -m eeg_cogcappro.reconstruct \
  --mode atms_ensemble_train_nearest \
  --method auto \
  --data-dir image-eeg-data \
  --feature-cache cache/features_vitl.pt \
  --retrieval-logits results/multi_encoder_ensemble/retrieval_test_logits.pt \
  --output-dir recons/atms_multimodal_final \
  --image-size 256 \
  --topk 5
