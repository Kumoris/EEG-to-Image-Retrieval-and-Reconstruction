#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-image-eeg-data}"
FEATURE_CACHE="${FEATURE_CACHE:-cache/features_rn50.pt}"
CONFIG="${CONFIG:-configs/cogcappro_rn50.yaml}"

for seed in 0 1 2 3 4 5 6 7 8 9; do
  python -m eeg_cogcappro.train \
    --config "$CONFIG" \
    --data-dir "$DATA_DIR" \
    --feature-cache "$FEATURE_CACHE" \
    --seed "$seed" \
    --output-dir "runs/seed${seed}"
  python -m eeg_cogcappro.eval_retrieval \
    --data-dir "$DATA_DIR" \
    --feature-cache "$FEATURE_CACHE" \
    --ckpt "runs/seed${seed}/best.pt" \
    --split test \
    --output "results/seed${seed}_test.json"
done
