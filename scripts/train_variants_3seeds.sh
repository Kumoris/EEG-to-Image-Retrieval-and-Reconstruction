#!/usr/bin/env bash
set -euo pipefail

BASE="/hpc2hdd/JH_DATA/jhai_data/dsaa2012_031/project_codex"
cd "$BASE"
source /hpc2hdd/home/dsaa2012_031/miniconda3/etc/profile.d/conda.sh
conda activate eeg

CACHE="cache/features_multi.pt"

run_variant() {
    local NAME="$1" CONFIG="$2" CKEY="$3"
    for SEED in 0 1 2; do
        echo "=== $NAME seed=$SEED ===" "$(date)"
        python3 -m eeg_cogcappro.train_atms \
            --config "$CONFIG" \
            --data-dir image-eeg-data \
            --feature-cache "$CACHE" \
            --feature-key "$CKEY" \
            --seed "$SEED" \
            --output-dir "runs/${NAME}_seed${SEED}" \
            --device cuda

        echo "=== Eval $NAME seed=$SEED ===" "$(date)"
        python3 -m eeg_cogcappro.eval_atms \
            --data-dir image-eeg-data \
            --feature-cache "$CACHE" \
            --feature-key "$CKEY" \
            --ckpt "runs/${NAME}_seed${SEED}/best.pt" \
            --split test \
            --tta-n 5 \
            --output "results/${NAME}_seed${SEED}_test_tta5.json" \
            --device cuda
    done
}

run_variant "conformer_ch_vitl"  "configs/atms_conformer_vitl.yaml"  "rn50_feature"
run_variant "deep_atms_rn50"     "configs/atms_deep_vitl.yaml"      "rn50_feature"

echo "=== All done ===" "$(date)"
