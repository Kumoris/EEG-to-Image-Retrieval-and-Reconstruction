#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-python3}"

for mod in text depth edge; do
  for seed in 0 1 2 3 4 5 6 7 8 9; do
    "$PYTHON" -m eeg_cogcappro.train_atms \
      --config configs/atms_vitl.yaml \
      --data-dir image-eeg-data \
      --feature-cache "cache/features_vitl_${mod}.pt" \
      --seed "${seed}" \
      --output-dir "runs/atms_${mod}_vitl_seed${seed}" \
      --device cuda \
      --save-last-as-best

    "$PYTHON" -m eeg_cogcappro.eval_atms \
      --data-dir image-eeg-data \
      --feature-cache "cache/features_vitl_${mod}.pt" \
      --ckpt "runs/atms_${mod}_vitl_seed${seed}/best.pt" \
      --split test \
      --output "results/atms_${mod}_vitl_seed${seed}_test_tta0.json" \
      --tta-n 0 \
      --device cuda
  done
done
