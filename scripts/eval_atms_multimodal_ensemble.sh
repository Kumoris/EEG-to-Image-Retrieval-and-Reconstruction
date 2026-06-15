#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-python3}"

"$PYTHON" scripts/ensemble_retrieval.py \
  --output-dir results/atms_multimodal_ensemble \
  --split test \
  --normalize row_zscore \
  --modality "image=results/atms_vitl_seed*_test_tta0.logits.pt" \
  --modality "depth=results/atms_depth_vitl_seed*_test_tta0.logits.pt" \
  --modality "edge=results/atms_edge_vitl_seed*_test_tta0.logits.pt" \
  --modality "fusion=results/atms_fusion_vitl_seed0_test.logits.pt" \
  --weights image=0.5 \
  --weights depth=0.2 \
  --weights edge=0.2 \
  --weights fusion=0.1
