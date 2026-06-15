#!/usr/bin/env bash
set -euo pipefail

ROOT="recons/experiments"
RESULTS="results/reconstruction_experiments"
RETRIEVAL_LOGITS="results/multi_encoder_ensemble/retrieval_test_logits.pt"
PYTHON="${PYTHON:-python3}"
mkdir -p "$ROOT" "$RESULTS"

BASELINE="$ROOT/atms_ensemble_train_nearest_top5"
mkdir -p "$BASELINE"
cp recons/atms_multimodal_final/*.png "$BASELINE"/
cp recons/atms_multimodal_final/manifest.csv "$BASELINE"/ 2>/dev/null || true
cp recons/atms_multimodal_final/summary.json "$BASELINE"/ 2>/dev/null || true
"$PYTHON" -m eeg_cogcappro.eval_reconstruction_official \
  --fake-dir "$BASELINE" \
  --output "$RESULTS/atms_ensemble_train_nearest_top5.json" \
  --metrics requested \
  --batch-size 32 \
  --device auto \
  --allow-open-clip-fallback

METHODS=(
  train_nearest_top1
  train_nearest_rerank_topk
  concept_train_nearest
  postprocess_sharp_color
  diffusion_prompt
  diffusion_img2img_train_source
)

for method in "${METHODS[@]}"; do
  out_dir="$ROOT/$method"
  "$PYTHON" -m eeg_cogcappro.reconstruct_experiments \
    --method "$method" \
    --data-dir image-eeg-data \
    --feature-cache cache/features_vitl.pt \
    --retrieval-logits "$RETRIEVAL_LOGITS" \
    --output-dir "$out_dir" \
    --topk 10 \
    --train-candidates 25 \
    --image-size 256 \
    --diffusion-model stabilityai/sdxl-turbo \
    --diffusion-steps 2 \
    --guidance-scale 0.0 \
    --strength 0.8

  png_count=$(find "$out_dir" -maxdepth 1 -type f -name '*.png' | wc -l)
  if [ "$png_count" -eq 200 ]; then
    "$PYTHON" -m eeg_cogcappro.eval_reconstruction_official \
      --fake-dir "$out_dir" \
      --output "$RESULTS/$method.json" \
      --metrics requested \
      --batch-size 32 \
      --device auto \
      --allow-open-clip-fallback
  fi
done

"$PYTHON" scripts/select_best_reconstruction.py \
  --experiments-root "$ROOT" \
  --results-root "$RESULTS" \
  --output-dir recons/atms_multimodal_final_improved \
  --summary results/reconstruction_experiments_summary.json
