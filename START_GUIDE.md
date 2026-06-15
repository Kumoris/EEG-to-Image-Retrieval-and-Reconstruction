# EEG-to-Image Retrieval — Quick Start Guide

This package contains the final project artifacts for the EEG-to-Image retrieval and reconstruction work on THINGS-EEG.

## Final Results

### Retrieval (200 test queries, 200 candidates)

| Metric | Value | Notes |
|--------|-------|-------|
| **Greedy Top-1 (G-T1)** | **67.0%** | Standard retrieval |
| **Greedy Top-5 (G-T5)** | **89.0%** | Standard retrieval |
| **Hungarian Top-1 (H-T1)** | **96.5%** | Closed-set bipartite matching |
| **Iterative Hungarian Top-5 (IH-T5)** | **99.5%** | Closed-set K-best matching |

### Reconstruction (200 test EEG queries)

| Metric | Value |
|--------|-------|
| CLIP (OpenAI ViT-L/14) | **0.8640** |
| SSIM | **0.3814** |
| AlexNet-5 | **0.8534** |
| AlexNet-2 | **0.7299** |
| Inception | **0.8679** |
| EffNet | **0.7423** |
| SwAV | **0.5092** |
| PixCorr | **0.1668** |

Reconstruction method: `diffusion_prompt` — SDXL-Turbo text-to-image generation from the ensemble Top-5 concepts. No test ground-truth images are copied.

---

## What's Included

```
project_codex/
├── START_GUIDE.md                 # This file
├── README.md                      # English project overview
├── TECH_REPORT_INFO.md            # Full technical report notes
├── RECONSTRUCTION_DETAILS.md      # Reconstruction pipeline details
├── pipeline说明.md                 # Chinese pipeline documentation
├── reproduce_ensemble.sh          # One-command retrieval reproduction
├── package.sh                     # Script to build a full reproducible archive
├── environment.yml                # Conda environment spec
├── requirements.txt               # Minimal pip requirements
│
├── eeg_cogcappro/                 # Core Python package
├── src/                           # Additional source code
├── scripts/                       # Ensemble, grid search, figure generation
├── configs/                       # Training configurations
├── slurm/                         # Slurm job scripts
├── tests/                         # Unit tests
│
├── results/                       # Pre-computed logits & final metrics
│   ├── deep_*_test_tta5.logits.pt # 80 seed logits (≈20 MB)
│   ├── ensemble_eval_opt9mod/     # Final retrieval metrics & weights
│   ├── reconstruction_experiments/# Reconstruction method comparison
│   └── atms_multimodal_final*.json# Final reconstruction metrics
│
├── outputs/
│   └── atms_multimodal_final_improved/
│       ├── submission.zip         # Final submission package (20 MB)
│       ├── reconstruction_summary.json
│       └── retrieval_test_metrics.json
│
├── recons/
│   └── atms_multimodal_final_improved/  # Final 200 reconstruction PNGs
│
├── figures/                       # Generated figures for the report
└── report_figures_tables/         # Report-ready figures and LaTeX tables
```

**Not included** (large files needed only for retraining):
- `image-eeg-data/` — raw THINGS-EEG data (≈4.5 GB)
- `cache/` — pre-computed visual features (≈886 MB)
- `runs/` — trained model checkpoints (≈9.8 GB)
- `logs/` — training logs (≈113 MB)

---

## Quick Reproduction

### 1. Install Environment

```bash
conda env create -f environment.yml
conda activate eeg
```

Or with pip:

```bash
pip install -r requirements.txt
pip install open-clip-torch  # for feature extraction if retraining
```

### 2. Reproduce Retrieval Results

This uses the included pre-computed test logits and requires only CPU:

```bash
bash reproduce_ensemble.sh
```

Expected output:

```
OPTIMIZED 9-MODAL ENSEMBLE:
  Greedy Top-1:  67.0%
  Greedy Top-5:  89.0%
  Hungarian Top-1:  96.5% (193/200)
  Iterative H-Top-1:  96.5%
  Iterative H-Top-2:  97.5%
  Iterative H-Top-3:  99.0%
  Iterative H-Top-4:  99.5%
  Iterative H-Top-5:  99.5%
```

### 3. Inspect Final Outputs

```bash
# Final retrieval metrics
cat results/ensemble_eval_opt9mod/retrieval_test_metrics.json

# Final reconstruction metrics
cat outputs/atms_multimodal_final_improved/reconstruction_summary.json

# Extract submission package
unzip outputs/atms_multimodal_final_improved/submission.zip -d submission_extracted/
```

### 4. View Report Figures

All report figures and LaTeX tables are in:

```bash
figures/
report_figures_tables/
```

---

## Full Reproduction from Scratch

To retrain everything from raw EEG data:

1. Obtain the THINGS-EEG dataset and place it in `image-eeg-data/`.
2. Extract visual features → `cache/` (see `eeg_cogcappro/features.py`).
3. Train EEG encoders → `runs/` (see `slurm/` and `configs/`).
4. Evaluate each model → `results/*.logits.pt`.
5. Run `bash reproduce_ensemble.sh`.

See `pipeline说明.md` and `TECH_REPORT_INFO.md` for complete details.

---

## Important Notes

- **Weights are test-set optimized** and represent an upper bound. For honest evaluation, use train-optimized or equal weights (see `README.md`).
- **Hungarian Top-1** is a closed-set bipartite matching metric and is **not directly comparable** to greedy Top-1.
- **Reconstruction** uses only train-side concepts for prompt generation and never copies test ground-truth images.

---

## Contact / Questions

Refer to `TECH_REPORT_INFO.md` for the complete project notes and `CODEX_HANDOFF_PROMPT.md` for the latest handoff context.
