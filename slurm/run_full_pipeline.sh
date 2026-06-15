#!/bin/bash
#SBATCH -p i64m1tga40u
#SBATCH -o logs/step0to4_full_%j.out
#SBATCH -e logs/step0to4_full_%j.err
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -D /hpc2hdd/JH_DATA/jhai_data/dsaa2012_031/project_codex

# Full pipeline: Step 0 -> Step 1 -> Step 2 -> Step 4

echo "============================================="
echo "  Full Multi-Encoder Pipeline"
echo "  Started at $(date)"
echo "  Node: $(hostname)"
echo "============================================="

source /hpc2hdd/home/dsaa2012_031/miniconda3/etc/profile.d/conda.sh
conda activate eeg

PYTHON=python3
DATA_DIR="image-eeg-data"
FEATURE_CACHE="cache/features_multi.pt"
DEVICE="cuda"

mkdir -p cache results logs

# ===================== Step 0: Feature Extraction =====================
echo ""
echo "===== Step 0: Extract multi-backend features ====="
echo "Started at $(date)"

$PYTHON -m eeg_cogcappro.features multi \
    --data-dir "$DATA_DIR" \
    --backends RN50 ViT-B-32 dinov2_da2 sd_vae \
    --output-cache "$FEATURE_CACHE" \
    --batch-size 32 \
    --feature-dim 512 \
    --device "$DEVICE"

echo "Step 0 complete at $(date)"

# ===================== Step 1: Train Experts =====================
echo ""
echo "===== Step 1: Train ATM-S experts (3 seeds) ====="

for ENC_INFO in "rn50 configs/atms_rn50.yaml rn50_feature" \
                "vitb32 configs/atms_vitb32.yaml vit_b_32_feature" \
                "dinov2 configs/atms_dinov2.yaml dinov2_da2_feature" \
                "vae configs/atms_vae.yaml vae_feature"; do
    read -r ENCODER CONFIG KEY <<< "$ENC_INFO"
    for seed in 0 1 2; do
        echo "--- $ENCODER seed $seed --- ($(date))"
        $PYTHON -m eeg_cogcappro.train_atms \
            --config "$CONFIG" \
            --data-dir "$DATA_DIR" \
            --feature-cache "$FEATURE_CACHE" \
            --feature-key "$KEY" \
            --seed "$seed" \
            --output-dir "runs/atms_${ENCODER}_seed${seed}" \
            --device "$DEVICE" \
            --save-last-as-best

        $PYTHON -m eeg_cogcappro.eval_atms \
            --data-dir "$DATA_DIR" \
            --feature-cache "$FEATURE_CACHE" \
            --feature-key "$KEY" \
            --ckpt "runs/atms_${ENCODER}_seed${seed}/best.pt" \
            --split test \
            --output "results/atms_${ENCODER}_seed${seed}_test.json" \
            --device "$DEVICE"
    done
done

echo "Step 1 complete at $(date)"

# ===================== Step 2: Ensemble Evaluation =====================
echo ""
echo "===== Step 2: Multi-encoder ensemble evaluation ====="

for ENC_INFO in "rn50 rn50_feature" "vitb32 vit_b_32_feature" "dinov2 dinov2_da2_feature" "vae vae_feature"; do
    read -r ENCODER KEY <<< "$ENC_INFO"
    for seed in 0 1 2; do
        LOGITS="results/atms_${ENCODER}_seed${seed}_test.logits.pt"
        if [ ! -f "$LOGITS" ]; then
            CKPT="runs/atms_${ENCODER}_seed${seed}/best.pt"
            if [ -f "$CKPT" ]; then
                echo "Generating logits for ${ENCODER} seed${seed}..."
                $PYTHON -m eeg_cogcappro.eval_atms \
                    --data-dir "$DATA_DIR" \
                    --feature-cache "$FEATURE_CACHE" \
                    --feature-key "$KEY" \
                    --ckpt "$CKPT" \
                    --split test \
                    --output "results/atms_${ENCODER}_seed${seed}_test.json" \
                    --device "$DEVICE"
            fi
        fi
    done
done

echo "Running ensemble evaluation..."
mkdir -p results/multi_encoder_ensemble

$PYTHON scripts/ensemble_retrieval.py \
    --modality "image=results/atms_vitl_seed*_test_tta0.logits.pt" \
    --modality "depth=results/atms_depth_vitl_seed*_test_tta0.logits.pt" \
    --modality "edge=results/atms_edge_vitl_seed*_test_tta0.logits.pt" \
    --modality "rn50=results/atms_rn50_seed*_test.logits.pt" \
    --modality "vitb32=results/atms_vitb32_seed*_test.logits.pt" \
    --modality "dinov2=results/atms_dinov2_seed*_test.logits.pt" \
    --modality "vae=results/atms_vae_seed*_test.logits.pt" \
    --weights "image=0.35" --weights "depth=0.15" --weights "edge=0.15" --weights "rn50=0.10" --weights "vitb32=0.10" --weights "dinov2=0.10" --weights "vae=0.05" \
    --normalize row_zscore \
    --output-dir results/multi_encoder_ensemble/ \
    2>&1 || echo "ensemble_retrieval.py may need adjustment - continuing"

echo "Step 2 complete at $(date)"

# ===================== Step 4: VAE Reconstruction =====================
echo ""
echo "===== Step 4: VAE reconstruction pipeline ====="

VAE_CKPT="runs/atms_vae_seed0/best.pt"

if [ -f "$VAE_CKPT" ]; then
    echo "Training VAE latent projector..."
    $PYTHON -m eeg_cogcappro.train_vae_projector \
        --config configs/atms_vae.yaml \
        --data-dir "$DATA_DIR" \
        --feature-cache "$FEATURE_CACHE" \
        --feature-key vae_feature \
        --atm-ckpt "$VAE_CKPT" \
        --seed 0 \
        --output-dir "runs/vae_projector_seed0" \
        --device "$DEVICE"

    echo "Generating VAE reconstructions..."
    $PYTHON -m eeg_cogcappro.reconstruct_vae \
        --data-dir "$DATA_DIR" \
        --feature-cache "$FEATURE_CACHE" \
        --feature-key vae_feature \
        --vae-ckpt "$VAE_CKPT" \
        --projector-ckpt "runs/vae_projector_seed0/best.pt" \
        --output-dir "recons/vae_seed0" \
        --method vae_decode \
        --device "$DEVICE"
else
    echo "WARNING: VAE checkpoint not found at $VAE_CKPT, skipping reconstruction"
fi

echo ""
echo "============================================="
echo "  Pipeline complete at $(date)"
echo "============================================="