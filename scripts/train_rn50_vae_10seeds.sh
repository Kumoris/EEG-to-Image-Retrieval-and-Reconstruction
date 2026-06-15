#!/bin/bash
# Submit training + eval for rn50 and vae, seeds 3-9
# Expanding from 3 seeds to 10 seeds each
set -euo pipefail

echo "=== Submitting rn50 seeds 3-9 ==="
for seed in 3 4 5 6 7 8 9; do
    JOB_ID=$(sbatch -p i64m1tga40ue --gres=gpu:1 --job-name=rn50s${seed} \
        --output=logs/deep_rn50_seed${seed}.out \
        --error=logs/deep_rn50_seed${seed}.err \
        slurm/train_deep_ext.sh rn50 ${seed} | awk '{print $4}')
    echo "rn50 seed ${seed}: Job ID ${JOB_ID}"
done

echo ""
echo "=== Submitting vae seeds 3-9 ==="
for seed in 3 4 5 6 7 8 9; do
    JOB_ID=$(sbatch -p i64m1tga40ue --gres=gpu:1 --job-name=vaes${seed} \
        --output=logs/deep_vae_seed${seed}.out \
        --error=logs/deep_vae_seed${seed}.err \
        slurm/train_deep_ext.sh vae ${seed} | awk '{print $4}')
    echo "vae seed ${seed}: Job ID ${JOB_ID}"
done

echo ""
echo "=== After training, re-run ensemble ==="
echo "python scripts/ensemble_retrieval.py \\"
echo "    --modality deep_rn50='results/deep_rn50_seed*_test_tta5.logits.pt' \\"
echo "    --modality deep_vae='results/deep_vae_seed*_test_tta5.logits.pt' \\"
echo "    --modality depth='results/deep_vitl_depth_seed*_test_tta5.logits.pt' \\"
echo "    --modality edge='results/deep_vitl_edge_seed*_test_tta5.logits.pt' \\"
echo "    --normalize row_zscore --hungarian --hungarian-topk 10 \\"
echo "    --output-dir results/ensemble_4mod_10seed --split test --topk 5"