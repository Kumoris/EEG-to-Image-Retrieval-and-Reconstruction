#!/bin/bash
#SBATCH -p i64m1tga40u
#SBATCH -o logs/step2_ensemble_%j.out
#SBATCH -e logs/step2_ensemble_%j.err
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -D /hpc2hdd/JH_DATA/jhai_data/dsaa2012_031/project_codex

echo "=== Step 2: Multi-encoder ensemble evaluation ==="
echo "Job started at $(date)"
echo "Node: $(hostname)"

export PYTHONNOUSERSITE=0
export PATH="$HOME/.local/bin:$PATH"

DATA_DIR="image-eeg-data"
FEATURE_CACHE="cache/features_multi.pt"

# First, generate logits for new encoder experts if not already present
for ENCODER in rn50 vitb32 dinov2 vae; do
    case "$ENCODER" in
        rn50) KEY="rn50_feature" ;;
        vitb32) KEY="vit_b_32_feature" ;;
        dinov2) KEY="dinov2_da2_feature" ;;
        vae) KEY="vae_feature" ;;
    esac
    for seed in 0 1 2; do
        LOGITS="results/atms_${ENCODER}_seed${seed}_test.logits.pt"
        if [ ! -f "$LOGITS" ]; then
            CKPT="runs/atms_${ENCODER}_seed${seed}/best.pt"
            if [ -f "$CKPT" ]; then
                echo "Generating logits for ${ENCODER} seed${seed}..."
                python3 -m eeg_cogcappro.eval_atms \
                    --data-dir "$DATA_DIR" \
                    --feature-cache "$FEATURE_CACHE" \
                    --feature-key "$KEY" \
                    --ckpt "$CKPT" \
                    --split test \
                    --output "results/atms_${ENCODER}_seed${seed}_test.json" \
                    --device cuda
            else
                echo "WARNING: $CKPT not found, skipping"
            fi
        else
            echo "Logits already exist: $LOGITS"
        fi
    done
done

# Run ensemble evaluation
echo "Running multi-encoder ensemble evaluation..."
mkdir -p results/multi_encoder_ensemble

python3 scripts/ensemble_retrieval.py \
    --modality "image=results/atms_vitl_seed*_test_tta0.logits.pt" \
    --modality "depth=results/atms_depth_vitl_seed*_test_tta0.logits.pt" \
    --modality "edge=results/atms_edge_vitl_seed*_test_tta0.logits.pt" \
    --modality "rn50=results/atms_rn50_seed*_test.logits.pt" \
    --modality "vitb32=results/atms_vitb32_seed*_test.logits.pt" \
    --modality "dinov2=results/atms_dinov2_seed*_test.logits.pt" \
    --modality "vae=results/atms_vae_seed*_test.logits.pt" \
    --weights "image=0.35" "depth=0.15" "edge=0.15" "rn50=0.10" "vitb32=0.10" "dinov2=0.10" "vae=0.05" \
    --normalize row_zscore \
    --output-dir results/multi_encoder_ensemble/ \
    2>&1 || echo "ensemble_retrieval.py may need adjustment - continuing"

echo "=== Step 2 complete at $(date) ==="