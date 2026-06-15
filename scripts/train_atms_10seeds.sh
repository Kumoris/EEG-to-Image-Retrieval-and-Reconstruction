#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-image-eeg-data}"
FEATURE_CACHE="${FEATURE_CACHE:-cache/features_vitl.pt}"
CONFIG="${CONFIG:-configs/atms_vitl.yaml}"
PYTHON="${PYTHON:-python3}"

for seed in 0 1 2 3 4 5 6 7 8 9; do
  if [ ! -f "results/atms_vitl_seed${seed}_test_tta0.json" ]; then
    "$PYTHON" -m eeg_cogcappro.train_atms \
      --config "$CONFIG" \
      --data-dir "$DATA_DIR" \
      --feature-cache "$FEATURE_CACHE" \
      --seed "$seed" \
      --output-dir "runs/atms_vitl_seed${seed}" \
      --device cuda \
      --save-last-as-best
    "$PYTHON" -m eeg_cogcappro.eval_atms \
      --data-dir "$DATA_DIR" \
      --feature-cache "$FEATURE_CACHE" \
      --ckpt "runs/atms_vitl_seed${seed}/best.pt" \
      --split test \
      --output "results/atms_vitl_seed${seed}_test_tta0.json" \
      --tta-n 0 \
      --device cuda
  fi
done
