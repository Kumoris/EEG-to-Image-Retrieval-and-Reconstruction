#!/bin/bash
#SBATCH -p i64m1tga40u
#SBATCH -o logs/step1_train_%j.out
#SBATCH -e logs/step1_train_%j.err
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH -D /hpc2hdd/JH_DATA/jhai_data/dsaa2012_031/project_codex

echo "=== Step 1: Train ATM-S experts (3 seeds per encoder) ==="
echo "Job started at $(date)"
echo "Node: $(hostname)"

export PYTHONNOUSERSITE=0
export PATH="$HOME/.local/bin:$PATH"

mkdir -p results logs

DATA_DIR="image-eeg-data"
FEATURE_CACHE="cache/features_multi.pt"

if [ ! -f "$FEATURE_CACHE" ]; then
    echo "ERROR: $FEATURE_CACHE not found. Run step 0 first."
    exit 1
fi

# --- RN50 ---
for seed in 0 1 2; do
    echo "--- RN50 seed $seed --- ($(date))"
    python3 -m eeg_cogcappro.train_atms \
        --config configs/atms_rn50.yaml \
        --data-dir "$DATA_DIR" \
        --feature-cache "$FEATURE_CACHE" \
        --feature-key rn50_feature \
        --seed "$seed" \
        --output-dir "runs/atms_rn50_seed${seed}" \
        --device cuda \
        --save-last-as-best

    python3 -m eeg_cogcappro.eval_atms \
        --data-dir "$DATA_DIR" \
        --feature-cache "$FEATURE_CACHE" \
        --feature-key rn50_feature \
        --ckpt "runs/atms_rn50_seed${seed}/best.pt" \
        --split test \
        --output "results/atms_rn50_seed${seed}_test.json" \
        --device cuda
done

# --- ViT-B/32 ---
for seed in 0 1 2; do
    echo "--- ViT-B/32 seed $seed --- ($(date))"
    python3 -m eeg_cogcappro.train_atms \
        --config configs/atms_vitb32.yaml \
        --data-dir "$DATA_DIR" \
        --feature-cache "$FEATURE_CACHE" \
        --feature-key vit_b_32_feature \
        --seed "$seed" \
        --output-dir "runs/atms_vitb32_seed${seed}" \
        --device cuda \
        --save-last-as-best

    python3 -m eeg_cogcappro.eval_atms \
        --data-dir "$DATA_DIR" \
        --feature-cache "$FEATURE_CACHE" \
        --feature-key vit_b_32_feature \
        --ckpt "runs/atms_vitb32_seed${seed}/best.pt" \
        --split test \
        --output "results/atms_vitb32_seed${seed}_test.json" \
        --device cuda
done

# --- DINOv2-da2 ---
for seed in 0 1 2; do
    echo "--- DINOv2-da2 seed $seed --- ($(date))"
    python3 -m eeg_cogcappro.train_atms \
        --config configs/atms_dinov2.yaml \
        --data-dir "$DATA_DIR" \
        --feature-cache "$FEATURE_CACHE" \
        --feature-key dinov2_da2_feature \
        --seed "$seed" \
        --output-dir "runs/atms_dinov2_seed${seed}" \
        --device cuda \
        --save-last-as-best

    python3 -m eeg_cogcappro.eval_atms \
        --data-dir "$DATA_DIR" \
        --feature-cache "$FEATURE_CACHE" \
        --feature-key dinov2_da2_feature \
        --ckpt "runs/atms_dinov2_seed${seed}/best.pt" \
        --split test \
        --output "results/atms_dinov2_seed${seed}_test.json" \
        --device cuda
done

# --- VAE ---
for seed in 0 1 2; do
    echo "--- VAE seed $seed --- ($(date))"
    python3 -m eeg_cogcappro.train_atms \
        --config configs/atms_vae.yaml \
        --data-dir "$DATA_DIR" \
        --feature-cache "$FEATURE_CACHE" \
        --feature-key vae_feature \
        --seed "$seed" \
        --output-dir "runs/atms_vae_seed${seed}" \
        --device cuda \
        --save-last-as-best

    python3 -m eeg_cogcappro.eval_atms \
        --data-dir "$DATA_DIR" \
        --feature-cache "$FEATURE_CACHE" \
        --feature-key vae_feature \
        --ckpt "runs/atms_vae_seed${seed}/best.pt" \
        --split test \
        --output "results/atms_vae_seed${seed}_test.json" \
        --device cuda
done

echo "=== Step 1 complete at $(date) ==="