#!/usr/bin/env bash
set -euo pipefail

# Master pipeline for RN50 + ViT-b/32 + VAE + DINOv2_da2 multi-encoder approach
# Run each phase separately or the full pipeline at once.

DATA_DIR="${DATA_DIR:-image-eeg-data}"
DEVICE="${DEVICE:-cuda}"
FEATURE_CACHE="${FEATURE_CACHE:-cache/features_multi.pt}"

echo "====================================================================="
echo "  Multi-Encoder Pipeline: RN50 + ViT-B/32 + DINOv2_da2 + VAE"
echo "====================================================================="
echo ""
echo "DATA_DIR:   $DATA_DIR"
echo "DEVICE:     $DEVICE"
echo "FEATURE_CACHE: $FEATURE_CACHE"
echo ""

# ========== Phase 0: Feature Extraction ==========
echo "===== Phase 0: Extract multi-backend features ====="

if [ ! -f "$FEATURE_CACHE" ]; then
    echo "Feature cache not found. Extracting..."
    bash scripts/prepare_multi_features.sh
else
    echo "Feature cache already exists: $FEATURE_CACHE"
    echo "To re-extract, delete the cache and re-run."
fi

echo ""

# ========== Phase 1: Single-encoder baseline (3 seeds) ==========
echo "===== Phase 1: Train ATM-S experts for each encoder (3 seeds each) ====="
echo "This will train and evaluate 4 encoder types × 3 seeds = 12 models"
echo "Estimated time: ~3h on A100"
echo ""
echo "To run: bash scripts/train_multi_experts_3seeds.sh"
echo ""

# ========== Phase 1b: Review baseline results ==========
echo "===== Phase 1b: Review single-encoder baseline results ====="
echo ""
echo "After training, run:"
echo ""
echo "  # Check per-encoder results"
echo "  for enc in rn50 vitb32 dinov2 vae; do"
echo "    echo \"--- \$enc ---\""
echo "    for s in 0 1 2; do"
echo "      cat results/atms_\${enc}_seed\${s}_test.json 2>/dev/null || echo 'not found'"
echo "    done"
echo "  done"
echo ""

# ========== Phase 2: Two-by-two ensemble validation ==========
echo "===== Phase 2: Ensemble validation ====="
echo ""
echo "After reviewing baseline results, run ensemble evaluation:"
echo ""
echo "  bash scripts/eval_multi_ensemble.sh"
echo ""

# ========== Phase 3: Fusion model (optional) ==========
echo "===== Phase 3: Multi-encoder fusion model (optional) ====="
echo ""
echo "If 2-by-2 ensembles show improvement, train fusion model:"
echo "  (Create configs/atms_multi_fusion.yaml with all expert checkpoints)"
echo "  (Train ATMFusionEncoder with 8 modalities)"
echo ""

# ========== Phase 4: VAE reconstruction ==========
echo "===== Phase 4: VAE reconstruction ====="
echo ""
echo "To train VAE projector and generate reconstructions:"
echo "  bash scripts/run_vae_reconstruction.sh"
echo ""

# ========== Phase 5: 10-seed final training ==========
echo "===== Phase 5: 10-seed final training (after Phase 2 validation) ====="
echo ""
echo "If Phase 2 shows improvement over baseline, expand to 10 seeds:"
echo "  bash scripts/train_multi_experts_10seeds.sh"
echo ""
echo "====================================================================="
echo "  Pipeline setup complete!"
echo "  Start with Phase 0 (feature extraction)"
echo "====================================================================="