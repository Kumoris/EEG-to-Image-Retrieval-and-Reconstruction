#!/bin/bash
#SBATCH -p i64m1tga40u
#SBATCH -o logs/final_submission_%j.out
#SBATCH -e logs/final_submission_%j.err
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -D /hpc2hdd/JH_DATA/jhai_data/dsaa2012_031/project_codex

set -euo pipefail

echo "============================================="
echo "  Final multi-encoder submission pipeline"
echo "  Started at $(date)"
echo "  Node: $(hostname)"
echo "  Job ID: ${SLURM_JOB_ID:-unknown}"
echo "============================================="

if [ -f /hpc2hdd/home/dsaa2012_031/miniconda3/etc/profile.d/conda.sh ]; then
    source /hpc2hdd/home/dsaa2012_031/miniconda3/etc/profile.d/conda.sh
    conda activate eeg
fi

export PYTHON="${PYTHON:-python3}"
export DEVICE="${DEVICE:-cuda}"
NEW_ENCODER_SEEDS="${NEW_ENCODER_SEEDS:-0 1 2}"

mkdir -p cache results recons outputs logs

require_vitl_logits() {
    local missing=0
    for seed in 0 1 2 3 4 5 6 7 8 9; do
        for path in \
            "results/atms_vitl_seed${seed}_test_tta0.logits.pt" \
            "results/atms_depth_vitl_seed${seed}_test_tta0.logits.pt" \
            "results/atms_edge_vitl_seed${seed}_test_tta0.logits.pt"; do
            if [ ! -f "$path" ]; then
                echo "Missing required ViT-L source logits: $path"
                missing=1
            fi
        done
    done
    if [ "$missing" -ne 0 ]; then
        echo "ERROR: Final ensemble requires ViT-L image/depth/edge logits for seeds 0..9."
        echo "Generate them first on a GPU compute node by running the relevant ViT-L/modalities"
        echo "training scripts inside a Slurm GPU allocation, for example:"
        echo "  bash scripts/train_atms_10seeds.sh"
        echo "  bash scripts/train_atms_modalities_10seeds.sh"
        exit 1
    fi
}

missing_new_encoder_logits() {
    local missing=0
    for encoder in rn50 vitb32 dinov2 vae; do
        for seed in $NEW_ENCODER_SEEDS; do
            if [ ! -f "results/atms_${encoder}_seed${seed}_test.logits.pt" ]; then
                echo "Missing multi-encoder logits: results/atms_${encoder}_seed${seed}_test.logits.pt"
                missing=1
            fi
        done
    done
    return "$missing"
}

echo ""
echo "===== Step 1: Ensure feature caches ====="
if [ -f cache/features_multi.pt ]; then
    echo "Feature cache already exists: cache/features_multi.pt"
else
    bash scripts/prepare_multi_features.sh
fi
if [ -f cache/features_vitl.pt ]; then
    echo "Feature cache already exists: cache/features_vitl.pt"
else
    FEATURE_CACHE=cache/features_vitl.pt bash scripts/prepare_vitl_features.sh
fi

echo ""
echo "===== Step 2: Ensure multi-encoder expert logits ====="
require_vitl_logits
if missing_new_encoder_logits; then
    echo "Required multi-encoder logits already exist for seeds: $NEW_ENCODER_SEEDS"
else
    echo "Some multi-encoder logits are missing; running expert training/evaluation."
    bash scripts/train_multi_experts_10seeds.sh
fi

echo ""
echo "===== Step 3: Evaluate final multi-encoder ensemble ====="
bash scripts/eval_multi_ensemble.sh

echo ""
echo "===== Step 4: Generate final reconstructions ====="
bash scripts/reconstruct_atms_final.sh

echo ""
echo "===== Step 5: Evaluate reconstruction candidates ====="
bash scripts/eval_reconstruction_official_final.sh
bash scripts/run_reconstruction_experiments.sh

echo ""
echo "===== Step 6: Package and validate final submission ====="
bash scripts/package_improved_submission.sh
"$PYTHON" scripts/check_final_submission.py

echo ""
echo "============================================="
echo "  Final submission pipeline complete at $(date)"
echo "============================================="
