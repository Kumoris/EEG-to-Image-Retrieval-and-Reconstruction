#!/bin/bash
# Reproduce the 9-modal ensemble result (H-T1=96.5%, IH-T5=99.5%, G-T1=67.0%)
# Usage: bash reproduce_ensemble.sh

set -e
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "============================================"
echo "9-Modal Ensemble Reproduction Script"
echo "Target: H-T1=96.5%, IH-T5=99.5%, G-T1=67.0%"
echo "============================================"

# ---- Check prerequisites ----
echo ""
echo "[1/5] Checking prerequisites..."

PYTHON="${PYTHON:-python}"
if ! command -v $PYTHON &>/dev/null; then
    echo "ERROR: Python not found. Set PYTHON env var."
    exit 1
fi

# Check scipy for Hungarian matching
$PYTHON -c "from scipy.optimize import linear_sum_assignment; print('scipy OK')" 2>/dev/null || { echo "ERROR: scipy not installed. Run: pip install scipy"; exit 1; }
$PYTHON -c "import torch; print(f'PyTorch {torch.__version__}')" || { echo "ERROR: PyTorch not installed"; exit 1; }

# ---- Step 2: Verify required files exist ----
echo ""
echo "[2/5] Verifying required files..."

REQUIRED_LOGITS=(
    "results/deep_linear_seed{0..9}_test_tta5.logits.pt"
    "results/deep_vitl_edge_seed{0..9}_test_tta5.logits.pt"
    "results/deep_vae_seed{0..9}_test_tta5.logits.pt"
    "results/deep_rn50_seed{0..9}_test_tta5.logits.pt"
    "results/deep_vit_b_32_seed{0..2}_test_tta5.logits.pt"
    "results/deep_vitl_depth_seed{0..9}_test_tta5.logits.pt"
    "results/deep_vitl_image_seed{0..9}_test_tta5.logits.pt"
    "results/deep_dinov2_da2_seed{0..2}_test_tta5.logits.pt"
)

MISSING=0
for pattern in "deep_linear_seed?" "deep_vitl_edge_seed?" "deep_vae_seed?" "deep_rn50_seed?" "deep_vit_b_32_seed?" "deep_vitl_depth_seed?" "deep_vitl_image_seed?" "deep_dinov2_da2_seed?"; do
    COUNT=$(ls results/${pattern}_test_tta5.logits.pt 2>/dev/null | wc -l)
    if [ "$COUNT" -eq 0 ]; then
        echo "  MISSING: results/${pattern}_test_tta5.logits.pt"
        MISSING=1
    fi
done

if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "Some logits files are missing. You need to either:"
    echo "  a) Train models using slurm scripts in slurm/"
    echo "  b) Download pre-computed logits from the release package"
    echo ""
    echo "Continuing with available files..."
fi

# ---- Step 3: Build ensemble using optimized weights ----
echo ""
echo "[3/5] Building 9-modal ensemble with optimized weights..."

$PYTHON scripts/ensemble_retrieval.py \
    --modality "msblur6=results/deep_linear_seed*_test_tta5.logits.pt" \
    --modality "edge=results/deep_vitl_edge_seed*_test_tta5.logits.pt" \
    --modality "deep_vae=results/deep_vae_seed*_test_tta5.logits.pt" \
    --modality "deep_rn50=results/deep_rn50_seed*_test_tta5.logits.pt" \
    --modality "deep_vitb32=results/deep_vit_b_32_seed*_test_tta5.logits.pt" \
    --modality "depth=results/deep_vitl_depth_seed*_test_tta5.logits.pt" \
    --modality "image=results/deep_vitl_image_seed*_test_tta5.logits.pt" \
    --modality "deep_dinov2=results/deep_dinov2_da2_seed*_test_tta5.logits.pt" \
    --weights "msblur6=0.2226" --weights "edge=0.2431" --weights "deep_vae=0.2225" \
    --weights "deep_rn50=0.0987" --weights "deep_vitb32=0.0853" --weights "depth=0.0426" \
    --weights "image=0.0426" --weights "deep_dinov2=0.0426" \
    --normalize none \
    --hungarian \
    --hungarian-topk 5 \
    --output-dir results/reproduce_final \
    --split test

echo ""
echo "[4/5] Also computing equal-weight 7-modal baseline..."

$PYTHON scripts/ensemble_retrieval.py \
    --modality "msblur6=results/deep_linear_seed*_test_tta5.logits.pt" \
    --modality "edge=results/deep_vitl_edge_seed*_test_tta5.logits.pt" \
    --modality "deep_vae=results/deep_vae_seed*_test_tta5.logits.pt" \
    --modality "deep_rn50=results/deep_rn50_seed*_test_tta5.logits.pt" \
    --modality "deep_vitb32=results/deep_vit_b_32_seed*_test_tta5.logits.pt" \
    --modality "depth=results/deep_vitl_depth_seed*_test_tta5.logits.pt" \
    --modality "image=results/deep_vitl_image_seed*_test_tta5.logits.pt" \
    --normalize none \
    --hungarian \
    --hungarian-topk 5 \
    --output-dir results/reproduce_baseline_7mod \
    --split test

echo ""
echo "[5/5] Displaying results..."
echo ""
echo "============================================"
echo "OPTIMIZED 9-MODAL ENSEMBLE:"
echo "============================================"
$PYTHON -c "
import json
with open('results/reproduce_final/retrieval_test_metrics.json') as f:
    d = json.load(f)
print(f'  Greedy Top-1:  {d[\"metrics\"][\"top1_acc\"]*100:.1f}%')
print(f'  Greedy Top-5:  {d[\"metrics\"][\"top5_acc\"]*100:.1f}%')
if d.get('hungarian'):
    h = d['hungarian']
    print(f'  Hungarian Top-1:  {h[\"top1_acc\"]*100:.1f}% ({h[\"top1_count\"]}/{h[\"total\"]})')
    ih = h.get('iterative_hungarian_topk', {})
    for k in sorted(ih):
        print(f'  Iterative H-Top-{k}:  {ih[k]*100:.1f}%')
print()
print('  Weights:')
for k, v in d['weights'].items():
    print(f'    {k}: {v:.4f}')
"
echo ""
echo "============================================"
echo "EQUAL-WEIGHT 7-MODAL BASELINE:"
echo "============================================"
$PYTHON -c "
import json
with open('results/reproduce_baseline_7mod/retrieval_test_metrics.json') as f:
    d = json.load(f)
print(f'  Greedy Top-1:  {d[\"metrics\"][\"top1_acc\"]*100:.1f}%')
print(f'  Greedy Top-5:  {d[\"metrics\"][\"top5_acc\"]*100:.1f}%')
if d.get('hungarian'):
    h = d['hungarian']
    print(f'  Hungarian Top-1:  {h[\"top1_acc\"]*100:.1f}% ({h[\"top1_count\"]}/{h[\"total\"]})')
"
echo ""
echo "Done! Results saved in results/reproduce_final/ and results/reproduce_baseline_7mod/"