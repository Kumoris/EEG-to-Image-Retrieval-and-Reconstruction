#!/usr/bin/env bash
set -euo pipefail
BASE="/hpc2hdd/JH_DATA/jhai_data/dsaa2012_031/project_codex"
cd "$BASE"
source /hpc2hdd/home/dsaa2012_031/miniconda3/etc/profile.d/conda.sh
conda activate eeg

CACHE="cache/features_multi.pt"
SEED="${1:-0}"
EPOCHS="${2:-50}"

for CKEY in rn50_feature vit_b_32_feature dinov2_da2_feature vae_feature; do
    TAG="deep_${CKEY%_feature}"
    echo "=== $TAG seed=$SEED ===" "$(date)"
    python3 -m eeg_cogcappro.train_atms \
        --config configs/atms_deep_vitl.yaml \
        --data-dir image-eeg-data \
        --feature-cache "$CACHE" \
        --feature-key "$CKEY" \
        --seed "$SEED" \
        --epochs "$EPOCHS" \
        --output-dir "runs/deep_${CKEY%_feature}_seed${SEED}" \
        --device cuda

    echo "=== Eval $TAG seed=$SEED ===" "$(date)"
    python3 -m eeg_cogcappro.eval_atms \
        --data-dir image-eeg-data \
        --feature-cache "$CACHE" \
        --feature-key "$CKEY" \
        --ckpt "runs/deep_${CKEY%_feature}_seed${SEED}/best.pt" \
        --split test \
        --tta-n 5 \
        --output "results/deep_${CKEY%_feature}_seed${SEED}_test_tta5.json" \
        --device cuda
done
echo "=== All features done for seed=$SEED ===" "$(date)"
