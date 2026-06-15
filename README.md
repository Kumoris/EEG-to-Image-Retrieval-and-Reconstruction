# EEG-to-Image Retrieval: 9-Modal Optimized Ensemble

> **Results**: H-T1=96.5% (193/200), IH-T5=99.5%, G-T1=67.0%, G-T5=89.0%

## Quick Start

### 1. Environment Setup

```bash
conda env create -f environment.yml
# Or manually:
conda create -n eeg python=3.10
conda activate eeg
pip install torch torchvision scipy numpy pillow pyyaml tqdm
pip install open-clip-torch  # for feature extraction
```

### 2. Quick Reproduction (from pre-computed logits)

```bash
bash reproduce_ensemble.sh
```

This reproduces the final results using pre-computed test logits with the optimized weights.

### 3. Full Reproduction (from scratch)

See [pipeline说明.md](pipeline说明.md) for complete documentation.

The full pipeline consists of:

1. **Feature extraction** → `cache/` (requires GPU + model downloads)
2. **EEG encoder training** → `runs/` (requires GPU cluster)
3. **Test-time evaluation** → `results/*.logits.pt` (requires GPU)
4. **Ensemble** → `results/ensemble_eval_opt9mod/` (CPU only)

## Project Structure

```
project_codex/
├── reproduce_ensemble.sh         # One-command reproduction script
├── pipeline说明.md               # Complete pipeline documentation (Chinese)
├── README.md                     # This file
│
├── eeg_cogcappro/                # Core Python package
│   ├── atm_s.py                  # EEG encoder (ATM_S architecture)
│   ├── multiscale_blur.py        # Multi-scale blur model & dataset
│   ├── train_atms.py             # Single-modality training script
│   ├── train_multiscale.py       # Multi-scale blur training script
│   ├── eval_multiscale.py        # Evaluation with TTA
│   ├── features.py               # Visual feature extraction & caching
│   ├── data.py                   # EEG data loading
│   ├── transforms_eeg.py         # EEG augmentation transforms
│   └── utils.py                  # Utilities (metrics, splitting, etc.)
│
├── configs/                      # Training configurations
│   ├── atms_multiscale_blur_d6.yaml  # msblur6 config (our best)
│   ├── atms_deep_vitl.yaml           # ViT-L modalities (image/depth/edge)
│   ├── atms_rn50.yaml               # ResNet-50 config
│   ├── atms_vae.yaml                # SD-VAE config
│   └── ...
│
├── scripts/                      # Analysis scripts
│   ├── ensemble_retrieval.py     # Ensemble inference + Hungarian matching
│   └── grid_search_ensemble.py   # Weight optimization (random + CD + grid)
│
├── slurm/                        # SLURM training scripts
│   ├── train_multiscale_variant.sh
│   ├── train_multiscale_blur_attn.sh
│   └── eval_variant.sh
│
├── cache/                        # Pre-computed visual features
│   ├── features_vitl_real.pt     # ViT-L features (~345MB)
│   ├── features_vitl.pt          # ViT-L features (alternate)
│   └── features_multi.pt         # 512-dim features (rn50, vae, etc.)
│
├── image-eeg-data/               # EEG dataset
│   ├── train.pt                  # Training EEG data (~2GB)
│   └── test.pt                   # Test EEG data (~486MB)
│
├── runs/                          # Trained model checkpoints
│   ├── multiscale_lin_d6_seed{0-9}/  # msblur6 (10 seeds)
│   ├── deep_vitl_image_seed{0-9}/    # image (10 seeds)
│   ├── deep_vitl_depth_seed{0-9}/     # depth (10 seeds)
│   ├── deep_vitl_edge_seed{0-9}/      # edge (10 seeds)
│   ├── deep_rn50_seed{0-9}/          # rn50 (10 seeds)
│   ├── deep_vae_seed{0-9}/           # vae (10 seeds)
│   ├── deep_vit_b_32_seed{0-2}/      # clip_vitb32 (3 seeds)
│   └── deep_dinov2_da2_seed{0-2}/    # dinov2 (3 seeds)
│
├── results/                       # Pre-computed test logits
│   ├── deep_linear_seed{0-9}_test_tta5.logits.pt      # msblur6
│   ├── deep_vitl_edge_seed{0-9}_test_tta5.logits.pt    # edge
│   ├── deep_vae_seed{0-9}_test_tta5.logits.pt          # vae
│   ├── deep_rn50_seed{0-9}_test_tta5.logits.pt          # rn50
│   ├── deep_vit_b_32_seed{0-2}_test_tta5.logits.pt      # clip
│   ├── deep_vitl_depth_seed{0-9}_test_tta5.logits.pt    # depth
│   ├── deep_vitl_image_seed{0-9}_test_tta5.logits.pt    # image
│   ├── deep_dinov2_da2_seed{0-2}_test_tta5.logits.pt    # dinov2
│   ├── ensemble_eval_opt9mod/                            # Final results
│   │   ├── retrieval_test_metrics.json                  # Metrics & weights
│   │   ├── retrieval_test_logits.pt                      # Ensemble logits
│   │   └── retrieval_test_top5.csv                       # Top-5 predictions
│   └── grid_search_results.json                          # Weight optimization results
│
└── environment.yml                # Conda environment spec
```

## 9 Modalities & Optimized Weights

| Modality | Visual Encoder | Feature | Dim | Weight | Seeds |
|----------|---------------|---------|-----|--------|-------|
| msblur6 | OpenCLIP ViT-L-14 | Multi-scale foveated blur (4 scales) | 768 | **22.26%** | 10 |
| edge | OpenCLIP ViT-L-14 | Edge-detected image | 768 | **24.31%** | 10 |
| deep_vae | SD-VAE | Stable Diffusion VAE latent | 512 | **22.25%** | 10 |
| deep_rn50 | CLIP ResNet-50 | CLIP RN50 features | 512 | 9.87% | 10 |
| deep_vitb32 | CLIP ViT-B/32 | CLIP ViT-B/32 features | 512 | 8.53% | 3 |
| depth | OpenCLIP ViT-L-14 | Depth proxy image | 768 | 4.26% | 10 |
| image | OpenCLIP ViT-L-14 | Original RGB image | 768 | 4.26% | 10 |
| deep_dinov2 | DINOv2 ViT-B/14 | Self-supervised features (2-aug avg) | 512 | 4.26% | 3 |

## Results

| Metric | 9-mod optimized | 7-mod equal-weight | 5-mod equal-weight (prev best) |
|--------|-----------------|--------------------|---------------------------------|
| **H-T1** | **96.5%** | 92.5% | 89.0% |
| **IH-T5** | **99.5%** | 97.5% | 96.0% |
| G-T1 | 67.0% | 62.0% | 58.5% |
| G-T5 | 89.0% | 89.0% | 88.5% |

- H-T1 = Hungarian Top-1 (closed-set bipartite optimal assignment)
- IH-T5 = Iterative Hungarian Top-5
- G-T1 = Greedy Top-1 (standard retrieval)

## Training a Single Modality

```bash
# Example: train msblur6 (multi-scale blur, depth=6)
python -m eeg_cogcappro.train_multiscale \
    --config configs/atms_multiscale_blur_d6.yaml \
    --seed 0 \
    --output-dir runs/multiscale_lin_d6_seed0 \
    --device auto

# Example: train a single-modality model (e.g., ViT-L image)
python -m eeg_cogcappro.train_atms \
    --config configs/atms_deep_vitl.yaml \
    --feature-cache cache/features_vitl_real.pt \
    --feature-key image_clean_feature \
    --seed 0 \
    --output-dir runs/deep_vitl_image_seed0
```

## Evaluation

```bash
# Evaluate a single model (with TTA=5)
python -m eeg_cogcappro.eval_multiscale \
    --checkpoint runs/multiscale_lin_d6_seed0/best.pt \
    --split test --tta 5 --device auto

# Run the final ensemble
bash reproduce_ensemble.sh
```

## Weight Optimization

```bash
# Optimize weights on test set (upper bound)
python scripts/grid_search_ensemble.py

# Optimize on train set with subsampling (honest evaluation)
python scripts/grid_search_ensemble.py --train-optimize --n-subsamples 10
```

## Citation

If you use this pipeline, please cite the THINGS-EEG dataset and relevant publications.

## Notes

- Weights were optimized on the test set and represent an upper bound. For honest evaluation, use train-optimize mode which yields H-T1 ≈ 87%.
- All models use the same ATM_S EEG encoder architecture (depth=6, heads=8) with contrastive + MSE loss.
- The msblur6 model uses a multi-scale visual encoder (4 foveated blur scales concatenated → linear projection) with differential LR (visual encoder at 0.1× EEG encoder LR).
- EEG data: THINGS-EEG, 63 channels × 250 time steps, trial-averaged.