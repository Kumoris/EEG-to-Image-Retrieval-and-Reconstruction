#!/bin/bash
#SBATCH -p i64m1tga40u
#SBATCH -o logs/step4_vae_recon_%j.out
#SBATCH -e logs/step4_vae_recon_%j.err
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH -D /hpc2hdd/JH_DATA/jhai_data/dsaa2012_031/project_codex

echo "=== Step 4: VAE reconstruction pipeline ==="
echo "Job started at $(date)"
echo "Node: $(hostname)"

export PYTHONNOUSERSITE=0
export PATH="$HOME/.local/bin:$PATH"

DATA_DIR="image-eeg-data"
FEATURE_CACHE="cache/features_multi.pt"
SEED=0
VAE_CKPT="runs/atms_vae_seed0/best.pt"

if [ ! -f "$FEATURE_CACHE" ]; then
    echo "ERROR: $FEATURE_CACHE not found. Run step 0 first."
    exit 1
fi

if [ ! -f "$VAE_CKPT" ]; then
    echo "ERROR: $VAE_CKPT not found. Run step 1 first."
    exit 1
fi

# Step 4a: Train VAE LatentProjector
echo "=== Step 4a: Train VAE latent projector ==="
python3 -m eeg_cogcappro.train_vae_projector \
    --config configs/atms_vae.yaml \
    --data-dir "$DATA_DIR" \
    --feature-cache "$FEATURE_CACHE" \
    --feature-key vae_feature \
    --atm-ckpt "$VAE_CKPT" \
    --seed "$SEED" \
    --output-dir "runs/vae_projector_seed${SEED}" \
    --device cuda

# Step 4b: Generate VAE reconstructions
echo "=== Step 4b: Generate VAE reconstructions ==="
python3 -m eeg_cogcappro.reconstruct_vae \
    --data-dir "$DATA_DIR" \
    --feature-cache "$FEATURE_CACHE" \
    --feature-key vae_feature \
    --vae-ckpt "$VAE_CKPT" \
    --projector-ckpt "runs/vae_projector_seed${SEED}/best.pt" \
    --output-dir "recons/vae_seed${SEED}" \
    --method vae_decode \
    --device cuda

echo "=== Step 4 complete at $(date) ==="