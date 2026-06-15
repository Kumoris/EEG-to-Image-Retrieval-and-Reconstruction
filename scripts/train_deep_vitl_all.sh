#!/usr/bin/env bash
# Train Deep ATM-S for ViT-L image/depth/edge, 10 seeds each, eval with TTA=5
# Prerequisites: real ViT-L features extracted to cache/features_vitl_real.pt
set -euo pipefail

BASE="/hpc2hdd/JH_DATA/jhai_data/dsaa2012_031/project_codex"
cd "$BASE"
source /hpc2hdd/home/dsaa2012_031/miniconda3/etc/profile.d/conda.sh
conda activate eeg

CACHE="cache/features_vitl_real.pt"
CONFIG="configs/atms_deep_vitl.yaml"
EPOCHS=50
SEEDS="0 1 2 3 4 5 6 7 8 9"

for CKEY in image_clean_feature depth_feature edge_feature; do
    TAG="deep_vitl_${CKEY%_feature}"
    case "$CKEY" in
        image_clean_feature) SHORT="image" ;;
        depth_feature) SHORT="depth" ;;
        edge_feature) SHORT="edge" ;;
    esac

    for SEED in $SEEDS; do
        RUNDIR="runs/deep_vitl_${SHORT}_seed${SEED}"
        RESULT_JSON="results/deep_vitl_${SHORT}_seed${SEED}_test_tta5.json"
        RESULT_LOGITS="results/deep_vitl_${SHORT}_seed${SEED}_test_tta5.logits.pt"

        if [ -f "$RESULT_LOGITS" ]; then
            echo "SKIP $TAG seed=$SEED (already exists)"
            continue
        fi

        echo "=== Train $SHORT seed=$SEED ===" "$(date)"
        python3 -m eeg_cogcappro.train_atms \
            --config "$CONFIG" \
            --data-dir image-eeg-data \
            --feature-cache "$CACHE" \
            --feature-key "$CKEY" \
            --seed "$SEED" \
            --epochs "$EPOCHS" \
            --output-dir "$RUNDIR" \
            --device cuda

        echo "=== Eval $SHORT seed=$SEED ===" "$(date)"
        python3 -m eeg_cogcappro.eval_atms \
            --data-dir image-eeg-data \
            --feature-cache "$CACHE" \
            --feature-key "$CKEY" \
            --ckpt "$RUNDIR/best.pt" \
            --split test \
            --tta-n 5 \
            --output "$RESULT_JSON" \
            --device cuda
    done
done
echo "=== All Deep ViT-L done ===" "$(date)"
