#!/usr/bin/env bash
set -euo pipefail
BASE="/hpc2hdd/JH_DATA/jhai_data/dsaa2012_031/project_codex"
cd "$BASE"
source /hpc2hdd/home/dsaa2012_031/miniconda3/etc/profile.d/conda.sh
conda activate eeg

CACHE="cache/features_multi.pt"
SEED="${1:-0}"

echo "=== Deep ATM-S seed=$SEED ===" "$(date)"
python3 -m eeg_cogcappro.train_atms \
    --config configs/atms_deep_vitl.yaml \
    --data-dir image-eeg-data \
    --feature-cache "$CACHE" \
    --feature-key rn50_feature \
    --seed "$SEED" \
    --output-dir "runs/deep_atms_rn50_seed${SEED}" \
    --device cuda

echo "=== Eval seed=$SEED ===" "$(date)"
python3 -m eeg_cogcappro.eval_atms \
    --data-dir image-eeg-data \
    --feature-cache "$CACHE" \
    --feature-key rn50_feature \
    --ckpt "runs/deep_atms_rn50_seed${SEED}/best.pt" \
    --split test \
    --tta-n 5 \
    --output "results/deep_atms_rn50_seed${SEED}_test_tta5.json" \
    --device cuda

echo "=== Done ===" "$(date)"
