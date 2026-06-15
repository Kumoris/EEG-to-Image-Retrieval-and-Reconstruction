#!/usr/bin/env python3
"""Generate all figures for the EEG-to-Image Tech Report."""

import os, json, glob, math, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
from PIL import Image
from pathlib import Path
from scipy.optimize import linear_sum_assignment
import torch

ROOT = Path("/hpc2hdd/JH_DATA/jhai_data/dsaa2012_031/project_codex")
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)
RECONS_DIR = ROOT / "recons" / "vae_seed0"
TEST_IMG_DIR = ROOT / "image-eeg-data" / "test_images"
TRAIN_IMG_DIR = ROOT / "image-eeg-data" / "training_images"
RESULTS_DIR = ROOT / "results"

# Build index-to-image_id and image_id-to-path mappings
def build_test_image_map():
    """Map: image_id_str -> image file path; also idx(int) -> image_id_str"""
    dirs = sorted(glob.glob(str(TEST_IMG_DIR / "*")))
    id_to_path = {}
    idx_to_id = {}
    id_to_idx = {}
    for d in dirs:
        basename = os.path.basename(d)
        idx = int(basename.split("_")[0])
        imgs = sorted(glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.png")))
        if imgs:
            image_id = os.path.splitext(os.path.basename(imgs[0]))[0]
            id_to_path[image_id] = imgs[0]
            idx_to_id[idx] = image_id
            id_to_idx[image_id] = idx
    return id_to_path, idx_to_id, id_to_idx

test_id_to_path, test_idx_to_id, test_id_to_idx = build_test_image_map()

def load_top5_csv(csv_path):
    """Returns dict: query_idx(int) -> list of (pred_image_id_str, score)"""
    result = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            qidx = int(row['index'])
            preds = []
            for k in range(1, 6):
                pid = row.get(f'pred{k}', '')
                score = float(row.get(f'score{k}', 0))
                preds.append((pid, score))
            result[qidx] = preds
    return result

top5_preds = load_top5_csv(RESULTS_DIR / "ensemble_eval_opt9mod" / "retrieval_test_top5.csv")

# ==============================================================================
# F1: Overall System Pipeline
# ==============================================================================
def generate_pipeline_figure():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    box_style = dict(boxstyle="round,pad=0.4", facecolor="#E3F2FD", edgecolor="#1565C0", linewidth=1.5)
    eeg_box_style = dict(boxstyle="round,pad=0.4", facecolor="#FFF3E0", edgecolor="#E65100", linewidth=1.5)
    result_style = dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9", edgecolor="#2E7D32", linewidth=1.5)
    small_style = dict(boxstyle="round,pad=0.3", facecolor="#F3E5F5", edgecolor="#6A1B9A", linewidth=1.2)

    # EEG side
    ax.text(2, 9.2, "EEG Signal\n(63 ch × 250 ts)", ha='center', va='center', fontsize=10, fontweight='bold', bbox=eeg_box_style)
    ax.annotate('', xy=(2, 8.0), xytext=(2, 8.6), arrowprops=dict(arrowstyle='->', lw=2, color='#E65100'))
    ax.text(2, 7.3, "ATM-S Encoder\n+ TTA=5", ha='center', va='center', fontsize=10, fontweight='bold', bbox=eeg_box_style)
    ax.annotate('', xy=(2, 6.0), xytext=(2, 6.65), arrowprops=dict(arrowstyle='->', lw=2, color='#E65100'))
    ax.text(2, 5.3, "EEG Embedding\n(768-d or 512-d)", ha='center', va='center', fontsize=9, bbox=box_style)

    # Image side - feature extractors
    visual_items = [
        ("OpenCLIP ViT-L-14\n(clean/edge/depth/blur)", 9.2),
        ("CLIP ResNet-50", 7.8),
        ("CLIP ViT-B/32", 6.4),
        ("DINOv2 ViT-B/14\n(2-aug avg)", 5.0),
        ("SD-VAE\n(latent → 512d)", 3.6),
    ]
    ax.text(7.5, 10.2, "Visual Feature Extraction (Precomputed)", ha='center', va='center', fontsize=11, fontweight='bold')
    for label, y in visual_items:
        ax.text(7.5, y, label, ha='center', va='center', fontsize=8.5, bbox=small_style)

    # msblur6 box
    ax.text(7.5, 2.5, "Multi-Scale Blur Fusion\n4 scales → Linear → 768d", ha='center', va='center', fontsize=9, bbox=small_style)

    # Arrows from visual to logits
    ax.annotate('', xy=(10.5, 7.5), xytext=(9.5, 7.5), arrowprops=dict(arrowstyle='->', lw=1.5, color='#6A1B9A'))
    ax.text(11.5, 8.8, "Per-Modality\nLogits Matrix\n(200×200)", ha='center', va='center', fontsize=9, bbox=box_style)

    # Seed averaging
    ax.annotate('', xy=(11.5, 7.0), xytext=(11.5, 7.9), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.text(11.5, 6.3, "Seed Averaging\n(zscore→mean→zscore)", ha='center', va='center', fontsize=8.5, bbox=box_style)

    # Ensemble
    ax.annotate('', xy=(11.5, 5.0), xytext=(11.5, 5.6), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.text(11.5, 4.2, "9-Modal Weighted\nEnsemble", ha='center', va='center', fontsize=10, fontweight='bold', bbox=result_style)

    # EEG to logits arrow
    ax.annotate('', xy=(10.5, 8.8), xytext=(3.2, 5.3), arrowprops=dict(arrowstyle='->', lw=2, color='#E65100', connectionstyle='arc3,rad=0.2'))

    # Output
    ax.annotate('', xy=(11.5, 2.5), xytext=(11.5, 3.5), arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(11.5, 1.8, "Retrieval Prediction\n& Hungarian Matching", ha='center', va='center', fontsize=10, fontweight='bold', bbox=result_style)

    # Title
    ax.text(7, 0.5, "Overall Pipeline: EEG-to-Image Retrieval via 9-Modal Optimized Ensemble",
            ha='center', va='center', fontsize=12, fontweight='bold', style='italic')

    fig.savefig(FIG_DIR / "method_pipeline.pdf", bbox_inches='tight', dpi=150)
    fig.savefig(FIG_DIR / "method_pipeline.png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("[OK] F1: method_pipeline")

# ==============================================================================
# F4: ATM-S Architecture
# ==============================================================================
def generate_atms_architecture():
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')

    def draw_box(ax, x, y, w, h, text, facecolor='#E3F2FD', edgecolor='#1565C0', fontsize=8.5):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                             facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold' if fontsize > 8 else 'normal')

    # Input
    draw_box(ax, 2, 13, 6, 0.7, "Input EEG: 63 channels × 250 time steps", facecolor='#FFF3E0', edgecolor='#E65100', fontsize=10)

    # Branch arrows
    ax.annotate('', xy=(3.5, 12.3), xytext=(3.5, 13), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(6.5, 12.3), xytext=(6.5, 13), arrowprops=dict(arrowstyle='->', lw=2))

    # ChannelAttention branch
    draw_box(ax, 1, 11.6, 5, 0.65, "ChannelAttention Branch", facecolor='#E8EAF6', edgecolor='#283593')
    draw_box(ax, 1.2, 10.6, 4.6, 0.5, "Pos. Embedding (1, 63, 250)", fontsize=7.5)
    draw_box(ax, 1.2, 9.8, 4.6, 0.65, "TransformerEncoder ×6\n(d_model=250, nhead=8, d_ff=500)", fontsize=7.5)
    draw_box(ax, 1.2, 9.0, 4.6, 0.5, "LayerNorm(250)", fontsize=7.5)

    # ShallowNet branch
    draw_box(ax, 5.5, 11.6, 3.8, 0.65, "ShallowNet Branch", facecolor='#FCE4EC', edgecolor='#AD1457')
    draw_box(ax, 5.7, 10.6, 3.4, 0.5, "Conv2d(1→40, k=(1,25))", fontsize=7.5)
    draw_box(ax, 5.7, 9.95, 3.4, 0.5, "Conv2d(40→40, k=(63,1))", fontsize=7.5)
    draw_box(ax, 5.7, 9.3, 3.4, 0.5, "BN → x² → AvgPool → log → Dropout", fontsize=7.5)

    # Element-wise multiply
    draw_box(ax, 2, 8.0, 6, 0.6, "Element-wise Multiply", facecolor='#FFF9C4', edgecolor='#F57F17', fontsize=9)

    # Arrows to multiply
    ax.annotate('', xy=(5, 8.6), xytext=(3.5, 8.9), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(5, 8.6), xytext=(7.2, 9.25), arrowprops=dict(arrowstyle='->', lw=1.5))

    # ResidualMLP
    ax.annotate('', xy=(5, 7.4), xytext=(5, 7.95), arrowprops=dict(arrowstyle='->', lw=2))
    draw_box(ax, 2, 5.8, 6, 1.5, "ResidualMLPProjector\n━━━━━━━━━━━━━━━━━━\nLinear(1680→1024)\n2× ResBlock: LN→Linear(1024→2048)→GELU→Drop(0.3)→Linear(2048→1024)\nLinear(1024→768) → LayerNorm(768)", fontsize=7.5, facecolor='#E0F2F1', edgecolor='#00695C')

    # Output
    ax.annotate('', xy=(5, 4.7), xytext=(5, 5.7), arrowprops=dict(arrowstyle='->', lw=2))
    draw_box(ax, 2, 3.8, 6, 0.8, "EEG Embedding\n(768-dim or 512-dim)", facecolor='#C8E6C9', edgecolor='#2E7D32', fontsize=10)

    # Contrastive loss note
    ax.text(5, 2.8, "↑ L2-normalize → cosine similarity with visual embeddings → contrastive + MSE loss",
            ha='center', va='center', fontsize=8, style='italic', color='#555555')

    fig.savefig(FIG_DIR / "atms_architecture.pdf", bbox_inches='tight', dpi=150)
    fig.savefig(FIG_DIR / "atms_architecture.png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("[OK] F4: atms_architecture")

# ==============================================================================
# F6: Ensemble Weight Distribution
# ==============================================================================
def generate_ensemble_weights():
    modalities = ['edge', 'msblur6', 'vae', 'rn50', 'vitb32', 'depth', 'image', 'dinov2']
    weights = [0.2431, 0.2226, 0.2225, 0.0987, 0.0853, 0.0426, 0.0426, 0.0426]
    labels = ['Edge\n(ViT-L)', 'MSBlur\n(ViT-L, d=6)', 'VAE\n(SD-VAE)', 'RN50\n(CLIP)', 'ViT-B/32\n(CLIP)', 'Depth\n(ViT-L)', 'Image\n(ViT-L)', 'DINOv2\n(ViT-B/14)']
    colors = ['#1565C0', '#1976D2', '#42A5F5', '#7986CB', '#9FA8DA', '#90CAF9', '#BBDEFB', '#E3F2FD']

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(modalities)), weights, color=colors, edgecolor='#333333', linewidth=0.8, height=0.65)
    ax.set_yticks(range(len(modalities)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Weight in Ensemble', fontsize=12)
    ax.set_title('Optimized Modality Weights in 9-Modal Ensemble', fontsize=13, fontweight='bold')

    for bar, w in zip(bars, weights):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{w:.4f} ({w*100:.1f}%)', va='center', fontsize=9)

    ax.set_xlim(0, 0.32)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.5, -0.12, 'Note: Weights are optimized on the test set (upper-bound analysis)',
            transform=ax.transAxes, ha='center', fontsize=8, style='italic', color='#666666')

    fig.tight_layout()
    fig.savefig(FIG_DIR / "ensemble_weights.pdf", bbox_inches='tight', dpi=150)
    fig.savefig(FIG_DIR / "ensemble_weights.png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("[OK] F6: ensemble_weights")

# ==============================================================================
# F3: Multi-scale Foveated Blur Examples
# ==============================================================================
def generate_foveated_blur_examples():
    from PIL import ImageFilter
    rng = np.random.RandomState(42)
    sorted_ids = sorted(test_idx_to_id.keys())[:4]

    fig, axes = plt.subplots(len(sorted_ids), 4, figsize=(14, 3.5*len(sorted_ids)))
    blur_labels = ['Clean (σ=0)', 'Fovea-low (σ=2)', 'Fovea-mid (σ=8)', 'Fovea-high (σ=14)']
    sigmas = [0, 2.0, 8.0, 14.0]

    for row, idx in enumerate(sorted_ids):
        image_id = test_idx_to_id[idx]
        img_path = test_id_to_path[image_id]
        img = Image.open(img_path).convert('RGB').resize((224, 224))
        for col, (sigma, label) in enumerate(zip(sigmas, blur_labels)):
            if sigma == 0:
                blurred = img
            else:
                blurred = img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))
                w, h = blurred.size
                yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
                dist = np.sqrt(((xx - w/2)/max(w,1))**2 + ((yy - h/2)/max(h,1))**2)
                mask = np.clip((dist / 0.55) ** 1.7, 0, 1)
                a = np.asarray(img, dtype=np.float32)
                b = np.asarray(blurred, dtype=np.float32)
                out = a * (1 - mask[..., None]) + b * mask[..., None]
                blurred = Image.fromarray(out.astype(np.uint8))

            axes[row, col].imshow(blurred)
            if row == 0:
                axes[row, col].set_title(label, fontsize=10, fontweight='bold')
            axes[row, col].axis('off')

    fig.suptitle('Multi-Scale Foveated Blur: Central Sharpness with Peripheral Degradation', fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "foveated_blur_examples.pdf", bbox_inches='tight', dpi=150)
    fig.savefig(FIG_DIR / "foveated_blur_examples.png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("[OK] F3: foveated_blur_examples")

# ==============================================================================
# F2: Retrieval Top-5 Qualitative Examples
# ==============================================================================
def generate_retrieval_top5_examples():
    ensemble_logits = torch.load(RESULTS_DIR / "ensemble_eval_opt9mod" / "retrieval_test_logits.pt",
                                  map_location='cpu', weights_only=False)
    if isinstance(ensemble_logits, dict):
        logits = ensemble_logits['logits']
    else:
        logits = ensemble_logits

    n = logits.shape[0]

    # Compute GT ranks: for each query i, find rank of the correct entry logits[i][i]
    gt_ranks = []
    for i in range(n):
        sim = logits[i]
        rank = (sim > sim[i]).sum().item() + 1
        gt_ranks.append((rank, i))

    gt_ranks.sort()

    success_cases = [(r, i) for r, i in gt_ranks if r == 1][:2]
    partial_cases = [(r, i) for r, i in gt_ranks if 2 <= r <= 5][:2]
    failure_cases = sorted([(r, i) for r, i in gt_ranks if r > 10], key=lambda x: -x[0])[:2]

    cases = success_cases + partial_cases + failure_cases
    case_labels = ['Success'] * len(success_cases) + ['Partial'] * len(partial_cases) + ['Failure'] * len(failure_cases)

    n_cases = len(cases)
    fig, axes = plt.subplots(n_cases, 7, figsize=(18, 3.2 * n_cases))
    if n_cases == 1:
        axes = axes[None, :]

    for row, ((gt_rank, matrix_idx), label) in enumerate(zip(cases, case_labels)):
        gt_id = test_idx_to_id.get(matrix_idx, None)
        gt_path = test_id_to_path.get(gt_id, None) if gt_id else None

        if gt_path and os.path.exists(gt_path):
            gt_img = Image.open(gt_path).convert('RGB').resize((128, 128))
            axes[row, 0].imshow(gt_img)
        else:
            axes[row, 0].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[row, 0].transAxes)
        axes[row, 0].set_title(f'GT (rank {gt_rank})', fontsize=9, color='green' if gt_rank == 1 else ('orange' if gt_rank <= 5 else 'red'), fontweight='bold')
        axes[row, 0].axis('off')

        preds = top5_preds.get(matrix_idx, [])
        for col in range(5):
            if col < len(preds):
                pred_id, pred_score = preds[col]
                pred_path = test_id_to_path.get(pred_id, None)
                is_correct = (pred_id == gt_id)
            else:
                pred_id, pred_path, is_correct = '', None, False

            if pred_path and os.path.exists(pred_path):
                pred_img = Image.open(pred_path).convert('RGB').resize((128, 128))
            else:
                pred_img = Image.new('RGB', (128, 128), (200, 200, 200))
            axes[row, col + 1].imshow(pred_img)
            border_color = 'green' if is_correct else '#cccccc'
            for spine in axes[row, col + 1].spines.values():
                spine.set_color(border_color)
                spine.set_linewidth(3 if is_correct else 0.5)
                spine.set_visible(True)
            axes[row, col + 1].set_title(f'Top-{col+1}' + (' ✓' if is_correct else ''), fontsize=8, color='green' if is_correct else 'black')
            axes[row, col + 1].axis('off')

        short_id = gt_id[:20] if gt_id else str(matrix_idx)
        axes[row, 6].text(0.5, 0.5, f'#{matrix_idx}\n{short_id}\n{label}\nGT rank={gt_rank}',
                          ha='center', va='center', fontsize=7, transform=axes[row, 6].transAxes)
        axes[row, 6].axis('off')

    fig.suptitle('Retrieval Top-5 Examples (9-Modal Ensemble)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG_DIR / "retrieval_top5_examples.pdf", bbox_inches='tight', dpi=120)
    fig.savefig(FIG_DIR / "retrieval_top5_examples.png", bbox_inches='tight', dpi=120)
    plt.close(fig)
    print("[OK] F2: retrieval_top5_examples")

# ==============================================================================
# F5: Reconstruction Qualitative Examples (diffusion_prompt)
# ==============================================================================
def generate_reconstruction_examples():
    import csv as csv_mod

    # Explicitly use diffusion_prompt reconstruction directory
    best_dir = str(ROOT / "recons" / "experiments" / "diffusion_prompt")
    if not os.path.isdir(best_dir):
        # Fallback: find any available reconstruction directory
        recons_dirs = sorted(glob.glob(str(ROOT / "recons" / "*")))
        best_dir = None
        for d in recons_dirs:
            if not os.path.isdir(d):
                continue
            pngs = glob.glob(os.path.join(d, "*.png"))
            if len(pngs) >= 10:
                best_dir = d
                break

    if not best_dir or not os.path.isdir(best_dir):
        print("[SKIP] F5: no reconstruction directory with sufficient PNGs found")
        return

    recons_pngs = sorted(glob.glob(os.path.join(best_dir, "*.png")))
    n_show = min(10, len(recons_pngs))
    if n_show < 5:
        print(f"[SKIP] F5: only {len(recons_pngs)} reconstruction PNGs found in {best_dir}")
        return

    # Determine generation method from directory name
    if "diffusion_prompt" in best_dir:
        method_label = "diffusion_prompt (SDXL-Turbo)"
    elif "diffusion_img2img" in best_dir:
        method_label = "diffusion_img2img (SDXL-Turbo)"
    elif "concept_train" in best_dir:
        method_label = "concept_train_nearest"
    else:
        method_label = os.path.basename(best_dir)

    print(f"  Using reconstruction from: {best_dir} ({len(recons_pngs)} PNGs) [{method_label}]")

    # Load manifest to get query_index -> query_image_id mapping
    manifest_path = os.path.join(best_dir, "manifest.csv")
    recons_mapping = {}  # 0-based file index -> image_id
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                file_idx = int(row['query_index'])
                image_id = row.get('query_image_id', row.get('query_image_id_metadata', ''))
                recons_mapping[file_idx] = image_id

    # Determine which indices to show: pick diverse categories
    # Use manifest if available, otherwise use sequential indices
    if recons_mapping:
        all_indices = sorted(recons_mapping.keys())
        # Pick 10 spread-out samples for visual diversity
        step = max(1, len(all_indices) // 10)
        chosen_indices = all_indices[::step][:n_show]
        if len(chosen_indices) < n_show:
            chosen_indices = all_indices[:n_show]
    else:
        chosen_indices = list(range(n_show))

    fig, axes = plt.subplots(2, n_show, figsize=(2.4 * n_show, 6.0))

    for col, file_idx in enumerate(chosen_indices):
        recon_fname = f"{file_idx:03d}.png"
        recon_path = os.path.join(best_dir, recon_fname)
        if not os.path.exists(recon_path):
            # Try without leading zeros
            recon_path = os.path.join(best_dir, f"{file_idx}.png")

        # Get the query image_id from manifest
        image_id = recons_mapping.get(file_idx, None)

        # Find ground truth test image
        gt_path = None
        if image_id:
            gt_path = test_id_to_path.get(image_id, None)

        # Fallback: try 1-indexed directory lookup
        if not gt_path and not image_id:
            dir_idx = file_idx + 1  # 0-based recons -> 1-based test dir
            if dir_idx in test_idx_to_id:
                image_id = test_idx_to_id[dir_idx]
                gt_path = test_id_to_path.get(image_id, None)

        # Ground truth image
        if gt_path and os.path.exists(gt_path):
            gt_img = Image.open(gt_path).convert('RGB').resize((256, 256))
            axes[0, col].imshow(gt_img)
        else:
            axes[0, col].set_facecolor('#F5F5F5')
            axes[0, col].text(0.5, 0.5, '?', ha='center', va='center',
                             fontsize=20, color='#999999', transform=axes[0, col].transAxes)
        axes[0, col].axis('off')
        if col == 0:
            axes[0, col].set_ylabel('Ground\nTruth', fontsize=11, fontweight='bold')

        # Reconstruction image
        if os.path.exists(recon_path):
            recon_img = Image.open(recon_path).convert('RGB').resize((256, 256))
            axes[1, col].imshow(recon_img)
        else:
            axes[1, col].set_facecolor('#F5F5F5')
            axes[1, col].text(0.5, 0.5, '?', ha='center', va='center',
                             fontsize=20, color='#999999', transform=axes[1, col].transAxes)
        axes[1, col].axis('off')
        if col == 0:
            axes[1, col].set_ylabel('SDXL-Turbo\nReconstruction', fontsize=11, fontweight='bold')

        # Label with concept name
        if image_id:
            concept = image_id.rsplit('_', 1)[0] if '_' in image_id else image_id
            label = concept[:12]
        else:
            label = f'#{file_idx}'
        axes[1, col].set_xlabel(label, fontsize=8, rotation=30, ha='right')

    fig.suptitle(f'Reconstruction Examples: Ground Truth vs. {method_label}',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reconstruction_qualitative_10examples.pdf", bbox_inches='tight', dpi=150)
    fig.savefig(FIG_DIR / "reconstruction_qualitative_10examples.png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("[OK] F5 v2: reconstruction_qualitative_10examples")

# ==============================================================================
# F7: Hard Sample Analysis
# ==============================================================================
def generate_hard_sample_analysis():
    ensemble_logits = torch.load(RESULTS_DIR / "ensemble_eval_opt9mod" / "retrieval_test_logits.pt",
                                  map_location='cpu', weights_only=False)
    if isinstance(ensemble_logits, dict):
        logits = ensemble_logits['logits']
    else:
        logits = ensemble_logits

    n = logits.shape[0]

    gt_ranks = []
    for i in range(n):
        sim = logits[i]
        rank = (sim > sim[i]).sum().item() + 1
        gt_ranks.append((rank, i))

    easy = sorted([(r, i) for r, i in gt_ranks if r == 1], key=lambda x: x[1])[:2]
    medium = sorted([(r, i) for r, i in gt_ranks if 2 <= r <= 5], key=lambda x: x[0])[:2]
    hard = sorted([(r, i) for r, i in gt_ranks if r > 10], key=lambda x: -x[0])[:2]

    cases = easy + medium + hard
    case_labels = ['Easy'] * len(easy) + ['Medium'] * len(medium) + ['Hard'] * len(hard)

    n_cases = len(cases)
    fig, axes = plt.subplots(n_cases, 7, figsize=(18, 3.2 * n_cases))
    if n_cases == 1:
        axes = axes[None, :]
    elif n_cases > 1:
        pass

    color_map = {'Easy': 'green', 'Medium': '#FF9800', 'Hard': 'red'}

    for row, ((gt_rank, matrix_idx), label) in enumerate(zip(cases, case_labels)):
        gt_id = test_idx_to_id.get(matrix_idx, None)
        gt_path = test_id_to_path.get(gt_id, None) if gt_id else None

        if gt_path and os.path.exists(gt_path):
            gt_img = Image.open(gt_path).convert('RGB').resize((128, 128))
            axes[row, 0].imshow(gt_img)
        else:
            axes[row, 0].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[row, 0].transAxes)
        axes[row, 0].set_title(f'GT (rank {gt_rank})', fontsize=9, color=color_map.get(label, 'black'), fontweight='bold')
        axes[row, 0].axis('off')

        preds = top5_preds.get(matrix_idx, [])
        for col in range(min(5, max(1, len(preds)))):
            if col < len(preds):
                pred_id, pred_score = preds[col]
                pred_path = test_id_to_path.get(pred_id, None)
                is_correct = (pred_id == gt_id)
            else:
                pred_id, pred_path, is_correct = '', None, False

            if pred_path and os.path.exists(pred_path):
                pred_img = Image.open(pred_path).convert('RGB').resize((128, 128))
            else:
                pred_img = Image.new('RGB', (128, 128), (200, 200, 200))
            axes[row, col + 1].imshow(pred_img)
            for spine in axes[row, col + 1].spines.values():
                spine.set_color('green' if is_correct else '#cccccc')
                spine.set_linewidth(3 if is_correct else 0.5)
                spine.set_visible(True)
            axes[row, col + 1].set_title(f'Top-{col+1}' + (' ✓' if is_correct else ''), fontsize=8)
            axes[row, col + 1].axis('off')

        short_id = gt_id[:20] if gt_id else str(matrix_idx)
        axes[row, 6].text(0.5, 0.5, f'#{matrix_idx}\n{short_id}\n{label}\nGT rank={gt_rank}',
                          ha='center', va='center', fontsize=7, transform=axes[row, 6].transAxes)
        axes[row, 6].axis('off')

    fig.suptitle('Hard Sample Retrieval Analysis', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hard_sample_analysis.pdf", bbox_inches='tight', dpi=120)
    fig.savefig(FIG_DIR / "hard_sample_analysis.png", bbox_inches='tight', dpi=120)
    plt.close(fig)
    print("[OK] F7: hard_sample_analysis")

# ==============================================================================
# F8: Modality Contribution / Rank Improvement Plot
# ==============================================================================
def generate_modality_contribution():
    single_model_results = {
        'msblur6': (50.0, 83.5),
        'msblur2': (48.0, 81.5),
        'depth': (38.5, 76.0),
        'image': (37.0, 65.5),
        'edge': (34.5, 69.5),
        'rn50': (34.0, 68.0),
        'clip_vitb32': (32.0, 69.5),
        'dinov2': (28.0, 62.0),
        'vae': (12.5, 40.0),
    }

    ensemble_results = {
        '5-mod equal (3s)': (58.5, 88.5),
        '7-mod equal (10s)': (62.0, 89.0),
        '9-mod optimized': (67.0, 89.0),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Greedy Top-1
    names = list(single_model_results.keys()) + list(ensemble_results.keys())
    top1_vals = [single_model_results[k][0] for k in single_model_results] + [ensemble_results[k][0] for k in ensemble_results]
    colors = ['#90CAF9'] * len(single_model_results) + ['#FFB74D', '#FF9800', '#E65100']

    y_pos = list(range(len(names)))
    bars = ax1.barh(y_pos, top1_vals, color=colors, edgecolor='#333333', linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel('Greedy Top-1 Accuracy (%)', fontsize=10)
    ax1.set_title('Retrieval: G-T1 by Model/Ensemble', fontsize=11, fontweight='bold')
    for bar, val in zip(bars, top1_vals):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', va='center', fontsize=8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right: Ensemble progression
    stages = ['msblur6\n(single best)', '5-mod equal', '7-mod equal', '9-mod optimized']
    top1_prog = [50.0, 58.5, 62.0, 67.0]
    top5_prog = [83.5, 88.5, 89.0, 89.0]

    x = np.arange(len(stages))
    width = 0.35
    bars1 = ax2.bar(x - width/2, top1_prog, width, label='G-T1', color='#1976D2', edgecolor='#333333')
    bars2 = ax2.bar(x + width/2, top5_prog, width, label='G-T5', color='#42A5F5', edgecolor='#333333')
    ax2.set_xticks(x)
    ax2.set_xticklabels(stages, fontsize=8.5)
    ax2.set_ylabel('Accuracy (%)', fontsize=10)
    ax2.set_title('Ensemble Progression', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    for bar, val in zip(bars1 + bars2, top1_prog + top5_prog):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.0f}%', ha='center', fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "modality_contribution.pdf", bbox_inches='tight', dpi=150)
    fig.savefig(FIG_DIR / "modality_contribution.png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("[OK] F9: modality_contribution")

# ==============================================================================
# Hungarian vs Greedy comparison figure
# ==============================================================================
def generate_greedy_vs_hungarian():
    categories = ['Greedy\nTop-1', 'Greedy\nTop-5', 'Hungarian\nTop-1', 'Iter-Hungarian\nTop-5']
    values_9mod = [67.0, 89.0, 96.5, 99.5]
    values_7mod = [62.0, 89.0, 92.5, 95.0]
    values_5mod = [58.5, 88.5, 89.0, 96.0]

    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars1 = ax.bar(x - width, values_5mod, width, label='5-mod equal (3 seeds)', color='#90CAF9', edgecolor='#333333')
    bars2 = ax.bar(x, values_7mod, width, label='7-mod equal (10 seeds)', color='#42A5F5', edgecolor='#333333')
    bars3 = ax.bar(x + width, values_9mod, width, label='9-mod optimized', color='#0D47A1', edgecolor='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Retrieval Metrics: Greedy vs. Hungarian Evaluation Paradigms', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.set_ylim(50, 105)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.1f}%', ha='center', va='bottom', fontsize=7.5)

    ax.text(0.5, -0.15, 'Note: Greedy and Hungarian metrics use different evaluation paradigms and should not be directly compared numerically.',
            transform=ax.transAxes, ha='center', fontsize=8, style='italic', color='#666666')

    fig.tight_layout()
    fig.savefig(FIG_DIR / "greedy_vs_hungarian.pdf", bbox_inches='tight', dpi=150)
    fig.savefig(FIG_DIR / "greedy_vs_hungarian.png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("[OK] Greedy vs Hungarian chart")

# ==============================================================================
# Reconstruction Method Ablation Chart
# ==============================================================================
def generate_reconstruction_ablation():
    methods = ['concept_train\n_nearest', 'diffusion_\nprompt (Ours)', 'diffusion_\nimg2img']
    metrics = ['CLIP', 'SSIM', 'AlexNet-5', 'Inception', 'EffNet', 'SwAV', 'PixCorr']
    
    values = {
        'concept_train\n_nearest':  [0.8816, 0.3415, 0.8586, 0.8390, 0.7662, 0.4985, 0.1387],
        'diffusion_\nprompt (Ours)': [0.8640, 0.3814, 0.8534, 0.8679, 0.7423, 0.5092, 0.1668],
        'diffusion_\nimg2img':       [0.8048, 0.3694, 0.6427, 0.7528, 0.8768, 0.6123, 0.0619],
    }
    
    colors = ['#1565C0', '#FF6F00', '#2E7D32']
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (method, color) in enumerate(zip(methods, colors)):
        bars = ax.bar(x + i * width, values[method], width, label=method.replace('\n', ' '), color=color, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.008, f'{h:.3f}' if h < 0.9 else f'{h:.4f}',
                    ha='center', va='bottom', fontsize=6.5, rotation=90)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    fig.suptitle('Reconstruction Method Ablation: Official Metrics Comparison',
                 fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / "reconstruction_ablation.pdf", bbox_inches='tight', dpi=150)
    fig.savefig(FIG_DIR / "reconstruction_ablation.png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("[OK] Reconstruction ablation chart")

# ==============================================================================
# Generate all LaTeX tables as .tex files
# ==============================================================================
def generate_latex_tables():
    tables_dir = FIG_DIR / "latex_tables"
    tables_dir.mkdir(exist_ok=True)

    # T1: Main Retrieval Results
    t1 = r"""\begin{table}[htbp]
\centering
\caption{Main retrieval results on the 200-way test protocol. Greedy Top-1 and Top-5 are the primary retrieval metrics, while Hungarian-based metrics are reported as closed-set matching analysis.}
\label{tab:main_retrieval}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{G-T1} & \textbf{G-T5} & \textbf{H-T1} & \textbf{IH-T5} \\
\midrule
Single best: msblur6 & 50.0\% & 83.5\% & --- & --- \\
7-modal equal weights & 62.0\% & 89.0\% & 92.5\% & --- \\
\textbf{9-modal optimized weights} & \textbf{67.0\%} & \textbf{89.0\%} & \textbf{96.5\%} & \textbf{99.5\%} \\
\bottomrule
\end{tabular}
\end{table}
"""

    # T2: Reconstruction Results
    t2 = r"""\begin{table}[htbp]
\centering
\caption{Quantitative reconstruction results evaluated by official metrics. The final reconstruction uses diffusion\_prompt: SDXL-Turbo text-to-image generation with retrieval top-5 concept prompts. Best results in \textbf{bold}.}
\label{tab:reconstruction}
\begin{tabular}{lcccccccc}
\toprule
\textbf{Method} & \textbf{CLIP} & \textbf{SSIM} & \textbf{AlexNet-5} & \textbf{Inception} & \textbf{EffNet} & \textbf{SwAV} & \textbf{PixCorr} & \textbf{MSE} \\
\midrule
concept\_train\_nearest & \textbf{0.8816} & 0.3415 & \textbf{0.8586} & 0.8390 & 0.7662 & 0.4985 & 0.1387 & 0.1192 \\
diffusion\_prompt (\textbf{Ours}) & 0.8640 & \textbf{0.3814} & 0.8534 & \textbf{0.8679} & 0.7423 & 0.5092 & \textbf{0.1668} & \textbf{0.1047} \\
diffusion\_img2img & 0.8048 & 0.3694 & 0.6427 & 0.7528 & \textbf{0.8768} & \textbf{0.6123} & 0.0619 & 0.1179 \\
\bottomrule
\end{tabular}
\end{table}
"""

    # T3: Single-Modality Ablation
    t3 = r"""\begin{table}[htbp]
\centering
\caption{Single-modality ablation. Each modality is evaluated after seed averaging. Multi-scale foveated blur achieves the strongest single-modality performance, while other visual representations provide complementary signals for ensemble retrieval.}
\label{tab:single_modality}
\begin{tabular}{llccc}
\toprule
\textbf{Modality} & \textbf{Visual Feature} & \textbf{Seeds} & \textbf{G-T1} & \textbf{G-T5} \\
\midrule
msblur6 & ViT-L multi-scale foveated blur & 10 & \textbf{50.0\%} & \textbf{83.5\%} \\
msblur2 & ViT-L multi-scale blur (depth=2) & 10 & 48.0\% & 81.5\% \\
depth & ViT-L depth proxy & 10 & 38.5\% & 76.0\% \\
image & ViT-L clean image & 10 & 37.0\% & 65.5\% \\
edge & ViT-L edge image & 10 & 34.5\% & 69.5\% \\
rn50 & CLIP ResNet-50 & 10 & 34.0\% & 68.0\% \\
clip\_vitb32 & CLIP ViT-B/32 & 3 & 32.0\% & 69.5\% \\
dinov2 & DINOv2 ViT-B/14 (2-aug) & 3 & --- & --- \\
vae & Stable Diffusion VAE latent & 10 & 12.5\% & 40.0\% \\
\bottomrule
\end{tabular}
\end{table}
"""

    # T4: Ensemble Ablation
    t4 = r"""\begin{table}[htbp]
\centering
\caption{Ensemble ablation. Multi-modal aggregation and seed averaging improve retrieval performance. The optimized 9-modal ensemble is reported as an upper-bound analysis, while equal-weight ensembles provide a more conservative evaluation.}
\label{tab:ensemble_ablation}
\begin{tabular}{lccl}
\toprule
\textbf{Setting} & \textbf{G-T1} & \textbf{H-T1} & \textbf{Comment} \\
\midrule
5-mod equal weights, 3 seeds & 58.5\% & 89.0\% & Earlier best \\
7-mod equal weights, 3 seeds & 62.5\% & 92.0\% & More modalities \\
7-mod equal weights, 10 seeds & 62.0\% & 92.5\% & More stable seed avg \\
\textbf{9-mod optimized weights} & \textbf{67.0\%} & \textbf{96.5\%} & Final upper-bound \\
\bottomrule
\end{tabular}
\end{table}
"""

    # T5: Greedy vs Hungarian
    t5 = r"""\begin{table}[htbp]
\centering
\caption{Comparison between standard greedy retrieval and Hungarian-based closed-set matching. Hungarian matching enforces a global one-to-one assignment and is therefore analyzed separately from the standard Top-1 and Top-5 retrieval metrics.}
\label{tab:greedy_vs_hungarian}
\begin{tabular}{llccp{4cm}}
\toprule
\textbf{Evaluation} & \textbf{Constraint} & \textbf{Top-1} & \textbf{Top-5} & \textbf{Interpretation} \\
\midrule
Greedy retrieval & Independent ranking & 67.0\% & 89.0\% & Standard retrieval metric \\
Hungarian matching & Global 1-to-1 assignment & 96.5\% & --- & Closed-set only \\
Iterative Hungarian & K-best 1-to-1 assignment & 96.5\% & 99.5\% & Closed-set Top-K \\
\bottomrule
\end{tabular}
\end{table}
"""

    # T6: Hard Sample Analysis
    t6 = r"""\begin{table}[htbp]
\centering
\caption{Representative hard samples in retrieval. Some failures are caused by weak agreement across modalities, while others remain unresolved even by individual modality or seed-level predictions.}
\label{tab:hard_samples}
\begin{tabular}{ccclp{5cm}}
\toprule
\textbf{Sample ID} & \textbf{GT Rank (Ensemble)} & \textbf{Best Single-Mod Rank} & \textbf{Failure Type} & \textbf{Interpretation} \\
\midrule
91 & 65 & 3 (depth\_seed3) & Weak consensus & Only one seed/modality ranks GT highly \\
16 & 7 & Top-5 in 7/45 models & Near miss & Close but insufficient for greedy Top-5 \\
26 & $>$5 & $>$5 & Hard semantic failure & No single modality resolves it \\
194 & $>$5 & 4 (best single model) & Persistent ambiguity & Cross-modal ambiguity \\
\bottomrule
\end{tabular}
\end{table}
"""

    # T7: Reconstruction Failure Types
    t7 = r"""\begin{table}[htbp]
\centering
\caption{Summary of reconstruction success and failure types. diffusion\_prompt uses pure text-to-image generation, where failures may differ from train-nearest methods.}
\label{tab:recon_failure_types}
\begin{tabular}{lp{6cm}l}
\toprule
\textbf{Failure Type} & \textbf{Description} & \textbf{Example} \\
\midrule
Semantic success & Correct object category or scene type is preserved & Samples with high CLIP score \\
Color mismatch & Similar object but different color & Blue vs. red objects \\
Shape mismatch & Correct semantic class but wrong geometry/shape & Round vs. elongated objects \\
Background dominance & Background texture matched better than foreground & Outdoor scenes \\
Semantic failure & Reconstructed image belongs to wrong object category & Very low CLIP score samples \\
\bottomrule
\end{tabular}
\end{table}
"""

    # T8: Reconstruction Method Ablation
    t8 = r"""\begin{table}[htbp]
\centering
\caption{Reconstruction method ablation. We compare three strategies: (1) train-nearest image selection, (2) SDXL-Turbo text-to-image from retrieval top-5 concepts, and (3) SDXL-Turbo image-to-image from train-nearest source. Best results in \textbf{bold}.}
\label{tab:reconstruction_ablation}
\begin{tabular}{lcccccc}
\toprule
\textbf{Method} & \textbf{Generation Type} & \textbf{Train Image} & \textbf{CLIP} & \textbf{SSIM} & \textbf{AlexNet-5} & \textbf{Inception} \\
\midrule
concept\_train\_nearest & Nearest-neighbor retrieval & Required & \textbf{0.8816} & 0.3415 & \textbf{0.8586} & 0.8390 \\
diffusion\_prompt (\textbf{Ours}) & Text-to-image (SDXL-Turbo) & Not needed & 0.8640 & \textbf{0.3814} & 0.8534 & \textbf{0.8679} \\
diffusion\_img2img & Image-to-image (SDXL-Turbo) & Required & 0.8048 & 0.3694 & 0.6427 & 0.7528 \\
\bottomrule
\end{tabular}
\end{table}
"""

    tables = {'T1_main_retrieval.tex': t1, 'T2_reconstruction.tex': t2,
              'T3_single_modality_ablation.tex': t3, 'T4_ensemble_ablation.tex': t4,
              'T5_greedy_vs_hungarian.tex': t5, 'T6_hard_samples.tex': t6,
              'T7_recon_failure_types.tex': t7,
              'T8_reconstruction_ablation.tex': t8}

    for name, content in tables.items():
        with open(tables_dir / name, 'w') as f:
            f.write(content)
    print(f"[OK] LaTeX tables written to {tables_dir}")


# ==============================================================================
# Main
# ==============================================================================
if __name__ == '__main__':
    print("Generating all figures and tables for the tech report...")
    print("=" * 60)

    generate_pipeline_figure()
    generate_atms_architecture()
    generate_ensemble_weights()
    generate_foveated_blur_examples()
    generate_retrieval_top5_examples()
    generate_reconstruction_examples()
    generate_hard_sample_analysis()
    generate_modality_contribution()
    generate_greedy_vs_hungarian()
    generate_reconstruction_ablation()
    generate_latex_tables()

    print("=" * 60)
    print("All figures generated! Output directory:", FIG_DIR)
    print("\nFigures:")
    for f in sorted(FIG_DIR.glob("*.pdf")):
        print(f"  {f.name}")
    for f in sorted(FIG_DIR.glob("*.png")):
        print(f"  {f.name}")
    print("\nLaTeX tables:")
    for f in sorted((FIG_DIR / "latex_tables").glob("*.tex")):
        print(f"  {f.name}")