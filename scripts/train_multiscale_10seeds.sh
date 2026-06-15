#!/bin/bash
# Train multiscale_blur models with seeds 1-9 and evaluate all
# After seed 0 validation succeeded with G-T1=27.5%
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJ_DIR"

echo "=== Submitting multiscale_blur training for seeds 1-9 ==="

for seed in $(seq 1 9); do
    JOB_ID=$(sbatch -p i64m1tga40ue --gres=gpu:1 --job-name=ms${seed} \
        --output=logs/multiscale_seed${seed}.out \
        --error=logs/multiscale_seed${seed}.err \
        slurm/train_multiscale_blur.sh ${seed} | awk '{print $4}')
    echo "Seed ${seed}: Training Job ID ${JOB_ID}"
done

echo ""
echo "=== After all training completes, run evaluation: ==="
echo "for seed in \$(seq 0 9); do"
echo "    sbatch -p i64m1tga40ue --gres=gpu:1 --job-name=ems\${seed} \\"
echo "        --output=logs/eval_multiscale_seed\${seed}.out \\"
echo "        --error=logs/eval_multiscale_seed\${seed}.err \\"
echo "        slurm/eval_multiscale.sh \${seed} 5"
echo "done"
echo ""
echo "=== Then ensemble: ==="
echo "python scripts/ensemble_retrieval.py \\"
echo "    --modality multiscale='results/deep_multiscale_seed*_test_tta5.logits.pt' \\"
echo "    --modality deep_rn50='results/deep_rn50_seed*_test_tta5.logits.pt' \\"
echo "    --modality deep_vae='results/deep_vae_seed*_test_tta5.logits.pt' \\"
echo "    --modality depth='results/deep_vitl_depth_seed*_test_tta5.logits.pt' \\"
echo "    --modality edge='results/deep_vitl_edge_seed*_test_tta5.logits.pt' \\"
echo "    --normalize row_zscore --hungarian --hungarian-topk 10 \\"
echo "    --output-dir results/ensemble_5mod_ms --split test --topk 5"