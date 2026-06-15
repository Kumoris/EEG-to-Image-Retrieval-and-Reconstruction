#!/usr/bin/env bash
set -euo pipefail

# Full 10-seed training for all 4 new encoder experts
# Run AFTER Phase 1 validates each encoder is viable

DATA_DIR="${DATA_DIR:-image-eeg-data}"
DEVICE="${DEVICE:-cuda}"
FEATURE_CACHE="${FEATURE_CACHE:-cache/features_multi.pt}"
PYTHON="${PYTHON:-python3}"

declare -A ENCODER_KEYS
ENCODER_KEYS[rn50]="rn50_feature"
ENCODER_KEYS[vitb32]="vit_b_32_feature"
ENCODER_KEYS[dinov2]="dinov2_da2_feature"
ENCODER_KEYS[vae]="vae_feature"

declare -A ENCODER_CONFIGS
ENCODER_CONFIGS[rn50]="configs/atms_rn50.yaml"
ENCODER_CONFIGS[vitb32]="configs/atms_vitb32.yaml"
ENCODER_CONFIGS[dinov2]="configs/atms_dinov2.yaml"
ENCODER_CONFIGS[vae]="configs/atms_vae.yaml"

for ENCODER in rn50 vitb32 dinov2 vae; do
    KEY="${ENCODER_KEYS[$ENCODER]}"
    CONFIG="${ENCODER_CONFIGS[$ENCODER]}"
    echo "=== Training 10 seeds for $ENCODER (feature_key=$KEY) ==="

    for seed in $(seq 0 9); do
        CKPT_DIR="runs/atms_${ENCODER}_seed${seed}"
        RESULT="results/atms_${ENCODER}_seed${seed}_test.json"

        if [ -f "$RESULT" ]; then
            echo "  seed $seed already evaluated, skipping"
            continue
        fi

        "$PYTHON" -m eeg_cogcappro.train_atms \
            --config "$CONFIG" \
            --data-dir "$DATA_DIR" \
            --feature-cache "$FEATURE_CACHE" \
            --feature-key "$KEY" \
            --seed "$seed" \
            --output-dir "$CKPT_DIR" \
            --device "$DEVICE" \
            --save-last-as-best

        "$PYTHON" -m eeg_cogcappro.eval_atms \
            --data-dir "$DATA_DIR" \
            --feature-cache "$FEATURE_CACHE" \
            --feature-key "$KEY" \
            --ckpt "$CKPT_DIR/best.pt" \
            --split test \
            --output "$RESULT" \
            --device "$DEVICE"
    done
done

echo "=== All 10-seed training complete ==="
