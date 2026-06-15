#!/usr/bin/env bash
# Full pipeline: generate train logits → honest grid search → generate comparison report
# Can be submitted as a single Slurm job or run manually.
set -euo pipefail

BASE="/hpc2hdd/JH_DATA/jhai_data/dsaa2012_031/project_codex"
cd "$BASE"
source /hpc2hdd/home/dsaa2012_031/miniconda3/etc/profile.d/conda.sh
conda activate eeg

RESULTS_DIR="results"
EXPECTED_TRAIN_LOGITS=42

echo "================================================================"
echo "Pipeline: eval_train → grid_search_honest → comparison_report"
echo "Start: $(date)"
echo "================================================================"

# ---------------------------------------------------------------------------
# Phase 1: Generate train logits (skip if all exist)
# ---------------------------------------------------------------------------
EXISTING=$(find ${RESULTS_DIR} -maxdepth 1 -name '*_train_tta5.logits.pt' 2>/dev/null | wc -l)
echo "[Phase 1] Train logits: ${EXISTING}/${EXPECTED_TRAIN_LOGITS} exist"

if [ "$EXISTING" -lt "$EXPECTED_TRAIN_LOGITS" ]; then
    echo "[Phase 1] Running eval_all_train.sh ..."
    bash scripts/eval_all_train.sh
else
    echo "[Phase 1] All train logits present, skipping eval."
fi

EXISTING=$(find ${RESULTS_DIR} -maxdepth 1 -name '*_train_tta5.logits.pt' 2>/dev/null | wc -l)
if [ "$EXISTING" -lt "$EXPECTED_TRAIN_LOGITS" ]; then
    echo "[ERROR] Expected ${EXPECTED_TRAIN_LOGITS} train logits, found ${EXISTING}. Aborting."
    exit 1
fi

# ---------------------------------------------------------------------------
# Phase 2: Honest grid search (train-optimize, test-evaluate)
# ---------------------------------------------------------------------------
echo ""
echo "[Phase 2] Running honest grid search (train-optimize + Hungarian)..."
python3 scripts/grid_search_ensemble.py \
    --train-optimize --hungarian --n-subsamples 10

echo "[Phase 2] Running honest grid search (train-optimize, Greedy)..."
python3 scripts/grid_search_ensemble.py \
    --train-optimize --n-subsamples 10

# ---------------------------------------------------------------------------
# Phase 3: Generate comparison report
# ---------------------------------------------------------------------------
echo ""
echo "[Phase 3] Generating comparison report..."
python3 scripts/generate_comparison_report.py

echo ""
echo "================================================================"
echo "Pipeline complete: $(date)"
echo "================================================================"
echo ""
echo "Key output files:"
echo "  - results/grid_search_honest_hungarian_results.json  (Hungarian, train-opt)"
echo "  - results/grid_search_honest_greedy_results.json      (Greedy, train-opt)"
echo "  - results/grid_search_results.json                    (Hungarian, test-opt, upper bound)"
echo "  - results/comparison_report.json / .txt               (side-by-side comparison)"
