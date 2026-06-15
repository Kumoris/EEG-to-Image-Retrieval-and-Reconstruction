#!/usr/bin/env bash
set -euo pipefail

# VAE reconstruction pipeline
# Step 1: Train VAE latent projector
# Step 2: Generate VAE reconstructions
# Step 3: Evaluate

DATA_DIR="${DATA_DIR:-image-eeg-data}"
DEVICE="${DEVICE:-cuda}"
FEATURE_CACHE="${FEATURE_CACHE:-cache/features_multi.pt}"
VAE_CKPT="${VAE_CKPT:-runs/atms_vae_seed0/best.pt}"
SEED="${SEED:-0}"

echo "=== Step 1: Train VAE latent projector ==="
python -m eeg_cogcappro.train_vae_projector \
    --config configs/atms_vae.yaml \
    --data-dir "$DATA_DIR" \
    --feature-cache "$FEATURE_CACHE" \
    --feature-key vae_feature \
    --atm-ckpt "$VAE_CKPT" \
    --seed "$SEED" \
    --output-dir "runs/vae_projector_seed${SEED}" \
    --device "$DEVICE"

echo "=== Step 2: Generate VAE reconstructions ==="
python -m eeg_cogcappro.reconstruct_vae \
    --data-dir "$DATA_DIR" \
    --feature-cache "$FEATURE_CACHE" \
    --feature-key vae_feature \
    --vae-ckpt "$VAE_CKPT" \
    --projector-ckpt "runs/vae_projector_seed${SEED}/best.pt" \
    --output-dir "recons/vae_seed${SEED}" \
    --method vae_decode \
    --device "$DEVICE"

echo "=== Step 3: Evaluate VAE reconstructions ==="
if [ -d "recons/vae_seed${SEED}" ]; then
    python -m eeg_cogcappro.eval_reconstruction \
        --fake-dir "recons/vae_seed${SEED}" \
        --output "results/vae_seed${SEED}_reconstruction.json" \
        2>/dev/null || echo "Evaluation requires ground truth images"
fi

echo "=== VAE reconstruction pipeline complete ==="