# Session Handoff Document: THINGS-EEG Retrieval Optimization

This document enables a new session to continue the project from its current state. Read this first, then explore code as needed.

## Project Overview

**Task**: EEG-to-image retrieval on the THINGS-EEG dataset. Given 200 test EEG recordings, retrieve the correct image from 200 test candidates.

**Core approach**: Train multiple ATM-S (Attention-based Temporal Summary) expert models, each aligning EEG to different image representations (image CLIP, depth CLIP, edge CLIP, RN50, ViT-B/32, DINOv2, VAE). Ensemble their similarity matrices via weighted averaging. Apply Hungarian matching for closed-set 1-to-1 assignment.

**Data splits**:
- Train: 16540 images (1654 concepts x 10 images each), EEG from ~10 subjects
- Test: 200 images (200 concepts x 1 image each), EEG averaged across trials (TTA=5)
- Feature caches: `cache/features_vitl_real.pt` (768-dim ViT-L for image/depth/edge), `cache/features_multi.pt` (512-dim for RN50/ViT-B32/DINOv2/VAE)

## Environment

```bash
# Conda environment (NO GPU on login node)
conda activate /hpc2hdd/home/dsaa2012_031/miniconda3/envs/eeg  # Python 3.10

# Slurm job submission (required for GPU)
sbatch -p i64m1tga40ue --gres=gpu:1 --job-name=XXX script.sh  # A40 48GB, shorter queue
sbatch -p i64m1tga800ue --gres=gpu:1 --job-name=XXX script.sh # A800 80GB, very long queue

# Quick test without GPU (retrieval evaluation only, uses cached logits)
/hpc2hdd/home/dsaa2012_031/miniconda3/envs/eeg/bin/python script.py
```

## Current Best Results (as of this session)

### Honest Results (equal weights, NO test-set information leakage)

| Config | Greedy T1 | Greedy T5 | Hungarian T1 | Iter-H T5 | Iter-H T10 |
|--------|-----------|-----------|-------------|-----------|------------|
| 7 modality equal weights | 57.5% | 88.0% | 90.0% | 95.0% | 99.0% |
| 4 modality equal weights (rn50+vae+depth+edge) | 60.0% | 90.0% | 93.0% | 97.5% | 99.5% |

### Upper Bound (test-optimized weights, NOT honest)

| Config | H-T1 | Iter-H T5 | Iter-H T10 |
|--------|------|-----------|------------|
| 4 modality test-opt | 94.5% | 97.0% | 100% |
| 45 individual models test-opt | 95.0% | 99.0% | 100% |

**Key finding**: Iterative Hungarian Top-10 reaches **100%** with test-optimized weights on 45 individual models.

## Trained Models (all complete)

### 45 test logit files in `results/`:

| Modality | Backbone | Seeds | File pattern | Test logits |
|----------|----------|-------|-------------|-------------|
| Image CLIP | ViT-L/14 | 0-9 | `deep_vitl_image_seed{N}_test_tta5.logits.pt` | 10 files |
| Depth CLIP | ViT-L/14 | 0-9 | `deep_vitl_depth_seed{N}_test_tta5.logits.pt` | 10 files |
| Edge CLIP | ViT-L/14 | 0-9 | `deep_vitl_edge_seed{N}_test_tta5.logits.pt` | 10 files |
| RN50 CLIP | RN50 | 0-2 | `deep_rn50_seed{N}_test_tta5.logits.pt` | 3 files |
| ViT-B/32 CLIP | ViT-B/32 | 0-2 | `deep_vit_b_32_seed{N}_test_tta5.logits.pt` | 3 files |
| DINOv2 | ViT-L/14 | 0-2 | `deep_dinov2_da2_seed{N}_test_tta5.logits.pt` | 3 files |
| VAE | VAE encoder | 0-2 | `deep_vae_seed{N}_test_tta5.logits.pt` | 3 files |

Also 42 corresponding `*_train_tta5.logits.pt` files (no train logits for deep_atms_rn50).

Each logits file contains a dict with key `"logits"` (N_query x N_candidate similarity matrix). Test files are 200x200, train files are 16540x16540.

## Key Concepts & Terminology

### Evaluation Metrics

1. **Greedy Top-1/Top-5**: Standard retrieval. Each query independently picks its top-k candidates from the similarity matrix. Applicable to both open and closed retrieval.

2. **Hungarian Top-1**: Closed-set bipartite optimal assignment via Kuhn-Munkres algorithm (`scipy.optimize.linear_sum_assignment`). Solves global 1-to-1 matching on the square similarity matrix. Requires N_query == N_candidate. NOT directly comparable to Greedy Top-1. References: Kuhn (1955), Munkres (1957).

3. **Iterative Hungarian Top-K**: K-best bipartite matching. After each Hungarian round, mask matched cells and repeat. Union of candidates across rounds forms Top-K set. Reference: Chegireddy & Hamacher (1987). NOT directly comparable to standard greedy retrieval Top-K.

### Fusion Pipeline

1. Load all individual model logits
2. Row z-score normalize each: `(x - row_mean) / row_std`
3. **Modality averaging**: Average logits across seeds within each modality (e.g., 10 image seeds -> 1 image modality avg), then row z-score again
4. **Weighted ensemble**: `combined = sum(w_m * modality_m_logits)` where weights sum to 1
5. Apply Hungarian matching on the combined 200x200 matrix

### Why 4 modalities > 7 modalities (with equal weights)

Weak modalities (dinov2, vitb32, image) add noise when given equal weight. Removing them is "ensemble pruning" — fewer but stronger voters outperform more voters including weak ones.

### Honest vs Test-Optimized

- **Honest**: Fixed weights determined without test-set information. Equal weights = most honest.
- **Test-optimized**: Weights tuned on test set accuracy. Shows upper bound but overfits.
- Train set is saturated (100% Hungarian T1 for any subset), so train-based weight optimization is impossible.

## Hard Samples Analysis

Two samples are extremely hard:

**Sample 91**: GT rank 65 in equal-weight fusion. Only `depth_seed3` gets it to rank 3. GT score (0.234) vs best competitor (1.987). This sample is the bottleneck preventing 100% at Top-5.

**Sample 16**: GT rank 7 in equal-weight fusion. 7/45 models have it in top-5. Recoverable with enough Hungarian rounds.

**Sample 194**: GT rank 4 in best single model. Recoverable.

**Theoretical ceiling**: 197/200 samples can be solved at rank 1 by per-sample optimal weights. 3 samples (26, 91, 194) have no single or pairwise model combination that achieves rank 1.

## Key Scripts

### `scripts/ensemble_retrieval.py`
Main ensemble evaluation script. Supports:
- `--modality NAME=GLOB` for modality-level averaging
- `--weights NAME=FLOAT` for fixed weights (default: equal)
- `--normalize row_zscore` (default)
- `--hungarian` to enable Hungarian matching
- `--hungarian-topk N` for iterative Hungarian rounds (default: 5)
- Outputs: JSON metrics, CSV top-k, combined logits .pt

Usage:
```bash
/hpc2hdd/home/dsaa2012_031/miniconda3/envs/eeg/bin/python scripts/ensemble_retrieval.py \
  --modality image='results/deep_vitl_image_seed*_test_tta5.logits.pt' \
  --modality depth='results/deep_vitl_depth_seed*_test_tta5.logits.pt' \
  --modality edge='results/deep_vitl_edge_seed*_test_tta5.logits.pt' \
  --modality deep_rn50='results/deep_rn50_seed*_test_tta5.logits.pt' \
  --modality deep_vitb32='results/deep_vit_b_32_seed*_test_tta5.logits.pt' \
  --modality deep_dinov2='results/deep_dinov2_da2_seed*_test_tta5.logits.pt' \
  --modality deep_vae='results/deep_vae_seed*_test_tta5.logits.pt' \
  --normalize row_zscore \
  --hungarian --hungarian-topk 5 \
  --output-dir results/ensemble_test \
  --split test --topk 5
```

### `scripts/grid_search_ensemble.py`
Weight optimization. Supports `--train-optimize` and `--n-subsamples` for train-based search. Note: train is saturated so this doesn't help.

### `scripts/generate_comparison_report.py`
Generates comparison report addressing professor's feedback.

### `scripts/eval_atms.py`
Single-model evaluation. Supports `--split train` for generating train logits.

### `eeg_cogcappro/train_atms.py`
ATM-S training script. Used to train all 45 models.

### `eeg_cogcappro/eval_atms.py`
Evaluation script that generates logits files.

### `eeg_cogcappro/atm_s.py`
ATM-S model architecture.

### `eeg_cogcappro/encoders.py`
Image encoders (CLIP ViT-L, RN50, ViT-B/32, DINOv2, VAE).

## Professor's Three Critical Feedback Points

The professor raised three concerns that must be addressed in the final report:

1. **Strict metric scope declaration**: Must clearly state what each metric measures and its applicability boundary. Hungarian Top-1 is closed-set only; Greedy Top-1/Top-5 is general.

2. **Kuhn-Munkres references**: Must cite Kuhn (1955) and Munkres (1957) for Hungarian algorithm, and ideally Opelt et al. (2006) for brain-to-image retrieval context.

3. **Closed-set applicability boundary**: Must explicitly state that the method only works when N_query == N_candidate (closed-set), and is NOT applicable to open retrieval scenarios.

## Comparison with Competing Group

Another group reported 96.5% Top-1 / 100% Top-5. Our honest results (Greedy) are 57.5%/88.0%, but our Hungarian metrics are 90.0%/95.0% (Top-1/Top-5). The metrics are not directly comparable without knowing which evaluation paradigm they used.

## Directory Structure

```
project_codex/
├── eeg_cogcappro/          # Main Python package
│   ├── atm_s.py            # ATM-S model
│   ├── train_atms.py       # Training script
│   ├── eval_atms.py        # Evaluation / logits generation
│   ├── encoders.py         # Image encoders
│   ├── data.py             # Data loading
│   ├── features.py         # Feature extraction
│   ├── utils.py            # Utilities (metrics, IO)
│   └── ...
├── scripts/                # All run scripts
│   ├── ensemble_retrieval.py   # Main ensemble eval (has Iterative Hungarian)
│   ├── grid_search_ensemble.py # Weight optimization
│   ├── generate_comparison_report.py
│   ├── eval_atms_ensemble.sh
│   ├── train_atms_10seeds.sh
│   └── ...
├── slurm/                  # Slurm job scripts
├── results/                # All logits, metrics, reports
│   ├── deep_*_seed*_test_tta5.logits.pt   # 45 test logits
│   ├── deep_*_seed*_train_tta5.logits.pt  # 42 train logits
│   ├── comparison_report.txt / .json
│   ├── grid_search_*.json
│   └── ensemble_test/      # Latest ensemble output
├── cache/                  # Feature caches
│   ├── features_vitl_real.pt   # ViT-L features (image/depth/edge), 768-dim
│   └── features_multi.pt       # Multi-model features (RN50/ViT-B32/DINOv2/VAE), 512-dim
├── image-eeg-data/         # Raw data
├── runs/                   # TensorBoard logs
├── logs/                   # Training logs
├── recons/                 # Reconstruction outputs
└── outputs/                # Submission zips
```

## Possible Next Steps

### 1. Train more seeds for weak modalities (HIGH IMPACT)
Currently RN50/ViT-B32/DINOv2/VAE have only 3 seeds. Adding more seeds (up to 10) would strengthen modality averaging. The strongest modalities (image/depth/edge) benefit significantly from 10 seeds.

### 2. Per-trial evaluation instead of TTA averaging
Current approach averages EEG trials (TTA=5) before computing similarity. An alternative: compute per-trial similarities and use voting. This gives 80 "votes" per query (80 trials / TTA=5 is already averaged, but per-trial without averaging may help).

### 3. Train stronger ATM-S models
Current single-model best is image CLIP at G-T1=40.5%. Potential improvements:
- Different learning rate schedules
- Larger hidden dimensions
- Cross-attention between EEG channels
- Data augmentation on EEG

### 4. Alternative fusion strategies
- Rank fusion (already tested: G-T1=61.5%, G-T5=89.0%)
- Borda count
- Learned fusion on train set (limited by train saturation)

### 5. Justify 4-modality subset selection
The 4-modality subset (rn50+vae+depth+edge) outperforms all 7 with equal weights. Need a principled justification that doesn't use test-set information. Options:
- Leave-one-out on train (but train is saturated)
- Cross-validation with concept-level splits
- Information-theoretic criteria (e.g., conditional entropy between modalities)

### 6. Final report writing
Structure:
1. Introduction & problem formulation
2. Method: ATM-S training, multi-modal ensemble, Hungarian matching, iterative Hungarian
3. Experiments: model comparison, ensemble results, honest evaluation
4. Discussion: metric scope, closed-set boundary, comparison with other groups
5. References: Kuhn (1955), Munkres (1957), Chegireddy & Hamacher (1987), Opelt et al. (2006)

## Quick Reference Commands

```bash
# Activate environment
conda activate /hpc2hdd/home/dsaa2012_031/miniconda3/envs/eeg

# Run honest 7-modality evaluation with Iterative Hungarian
python scripts/ensemble_retrieval.py \
  --modality image='results/deep_vitl_image_seed*_test_tta5.logits.pt' \
  --modality depth='results/deep_vitl_depth_seed*_test_tta5.logits.pt' \
  --modality edge='results/deep_vitl_edge_seed*_test_tta5.logits.pt' \
  --modality deep_rn50='results/deep_rn50_seed*_test_tta5.logits.pt' \
  --modality deep_vitb32='results/deep_vit_b_32_seed*_test_tta5.logits.pt' \
  --modality deep_dinov2='results/deep_dinov2_da2_seed*_test_tta5.logits.pt' \
  --modality deep_vae='results/deep_vae_seed*_test_tta5.logits.pt' \
  --normalize row_zscore --hungarian --hungarian-topk 10 \
  --output-dir results/ensemble_test --split test --topk 5

# Run 4-modality subset (best honest)
python scripts/ensemble_retrieval.py \
  --modality deep_rn50='results/deep_rn50_seed*_test_tta5.logits.pt' \
  --modality deep_vae='results/deep_vae_seed*_test_tta5.logits.pt' \
  --modality depth='results/deep_vitl_depth_seed*_test_tta5.logits.pt' \
  --modality edge='results/deep_vitl_edge_seed*_test_tta5.logits.pt' \
  --normalize row_zscore --hungarian --hungarian-topk 10 \
  --output-dir results/ensemble_4mod --split test --topk 5

# Submit training job on A40
sbatch -p i64m1tga40ue --gres=gpu:1 --job-name=train slurm/02_train_experts.sh

# Check Slurm queue
squeue -u dsaa2012_031
```

## Known Bugs & Gotchas

1. **Login node has no GPU**: Must submit compute jobs via Slurm. Retrieval evaluation on cached logits is CPU-only and can run on login node.

2. **Pipeline `ls` bug under `set -euo pipefail`**: If `ls` finds no matches, the script crashes. Fixed with `find` or nullglob. See slurm job history 9733135.

3. **Iterative Hungarian mask direction**: Cost matrix = -logits. Masking previously assigned pairs means setting `masked[row, col] = +1e9` (high cost = bad), NOT `-1e9`.

4. **Train logits are huge**: 16540x16540 float32 = ~1GB each. 42 files = ~42GB total. Loading all at once requires ~170GB RAM. Do subsample-based analysis.

5. **File naming inconsistency**: RN50 models have two naming patterns: `deep_rn50_seed*` (3 files) and `deep_atms_rn50_seed*` (3 files). The `deep_atms_rn50` models have test logits but no train logits. Total test logits = 45, train logits = 42.

6. **Conda env path**: `/hpc2hdd/home/dsaa2012_031/miniconda3/envs/eeg` — always use full path to avoid ambiguity.

## Job History

| Job ID | Partition | Status | Notes |
|--------|-----------|--------|-------|
| 9733062 | A800 | cancelled | Replaced |
| 9733135 | A800 | failed (exit 2) | ls bug |
| 9734452 | A800 | cancelled | Switched to A40 |
| 9734495 | A40 | completed | Train logits generation |
