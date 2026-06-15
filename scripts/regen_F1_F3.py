#!/usr/bin/env python3
"""Regenerate F1 (pipeline) and F3 (foveated blur) with improved design."""

import os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle
import matplotlib.patheffects as pe
from PIL import Image, ImageFilter

ROOT = "/hpc2hdd/JH_DATA/jhai_data/dsaa2012_031/project_codex"
FIG_DIR = os.path.join(ROOT, "figures")
TEST_IMG_DIR = os.path.join(ROOT, "image-eeg-data", "test_images")

# Build test image map
def build_test_image_map():
    dirs = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*")))
    id_to_path = {}
    idx_to_id = {}
    for d in dirs:
        basename = os.path.basename(d)
        idx = int(basename.split("_")[0])
        imgs = sorted(glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.png")))
        if imgs:
            image_id = os.path.splitext(os.path.basename(imgs[0]))[0]
            id_to_path[image_id] = imgs[0]
            idx_to_id[idx] = image_id
    return id_to_path, idx_to_id

test_id_to_path, test_idx_to_id = build_test_image_map()

# ==============================================================================
# F1: Redesigned Pipeline Figure
# ==============================================================================
def generate_pipeline_v2():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10.5)
    ax.axis('off')
    
    # Color palette
    EEG_COLOR = '#FFF3E0'
    EEG_EDGE = '#E65100'
    VIS_COLOR = '#E8EAF6'
    VIS_EDGE = '#283593'
    PROC_COLOR = '#E3F2FD'
    PROC_EDGE = '#1565C0'
    ENS_COLOR = '#E8F5E9'
    ENS_EDGE = '#2E7D32'
    OUT_COLOR = '#FCE4EC'
    OUT_EDGE = '#C62828'
    BLUR_BOX = '#FFF8E1'
    BLUR_EDGE = '#F57F17'
    
    def draw_box(ax, x, y, w, h, text, fc, ec, fontsize=9, fw='bold', alpha=0.9, ls='-'):
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.15", facecolor=fc, edgecolor=ec,
                             linewidth=1.8, linestyle=ls, alpha=alpha, zorder=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                fontweight=fw, zorder=3)
    
    def draw_arrow(ax, x1, y1, x2, y2, color='#555', lw=2, style='->', connectionstyle='arc3,rad=0'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, lw=lw, color=color,
                                    connectionstyle=connectionstyle), zorder=1)
    
    # ---- Title ----
    ax.text(8, 10.2, 'EEG-to-Image Retrieval: 9-Modal Optimized Ensemble Pipeline',
            ha='center', va='center', fontsize=14, fontweight='bold',
            color='#1A237E',
            path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    
    # ---- Left Column: EEG Path ----
    # Input EEG
    draw_box(ax, 2.5, 9.0, 3.2, 0.65, 'EEG Signal (63 ch × 250 ts)', EEG_COLOR, EEG_EDGE, fontsize=10)
    draw_arrow(ax, 2.5, 8.6, 2.5, 8.05, EEG_EDGE, lw=2.5)
    
    # ATM-S with sub-diagram
    draw_box(ax, 2.5, 7.6, 3.4, 0.7, 'ATM-S EEG Encoder', EEG_COLOR, EEG_EDGE, fontsize=10.5, fw='bold')
    ax.text(2.5, 7.2, 'ChannelAttn × ShallowNet × MLP', ha='center', fontsize=7.5, style='italic', color='#BF360C')
    
    draw_arrow(ax, 2.5, 6.95, 2.5, 6.4, EEG_EDGE, lw=2.5)
    
    # TTA
    draw_box(ax, 2.5, 6.1, 2.8, 0.55, 'TTA=5 Augmentation\n& L2-Normalize', PROC_COLOR, PROC_EDGE, fontsize=8.5)
    draw_arrow(ax, 2.5, 5.8, 2.5, 5.25, EEG_EDGE, lw=2)
    
    # EEG embedding
    draw_box(ax, 2.5, 4.95, 2.6, 0.55, 'EEG Embedding\n(768-d or 512-d)', '#BBDEFB', '#1565C0', fontsize=9)
    
    # ---- Right Column: Visual Path ----
    # Section header
    ax.text(9.5, 9.6, 'Visual Feature Extraction (Precomputed)', ha='center', fontsize=11,
            fontweight='bold', color='#283593')
    
    # Visual encoders in two sub-columns
    # Left sub-column: ViT-L based
    draw_box(ax, 7.8, 8.8, 2.8, 0.55, 'OpenCLIP ViT-L/14', VIS_COLOR, VIS_EDGE, fontsize=9)
    
    vis_features_l = [
        '● clean image (768d)',
        '● foveated blur ×3 (768d)',
        '● edge image (768d)',
        '● depth proxy (768d)',
    ]
    for i, txt in enumerate(vis_features_l):
        ax.text(7.8, 8.3 - i * 0.38, txt, ha='center', fontsize=8, color='#37474F')
    
    # Right sub-column: others
    draw_box(ax, 11.5, 8.8, 2.6, 0.55, 'Other Encoders', VIS_COLOR, VIS_EDGE, fontsize=9)
    
    vis_features_r = [
        '● CLIP RN50 (512d)',
        '● CLIP ViT-B/32 (512d)',
        '● DINOv2 ViT-B/14 (512d)',
        '● SD-VAE latent (512d)',
    ]
    for i, txt in enumerate(vis_features_r):
        ax.text(11.5, 8.3 - i * 0.38, txt, ha='center', fontsize=8, color='#37474F')
    
    # Blur fusion box
    draw_box(ax, 7.8, 6.6, 3.2, 0.55, 'Multi-Scale Blur Fusion\n4 scales → Linear → 768d', BLUR_BOX, BLUR_EDGE, fontsize=8.5)
    
    # Arrow from ViT-L features to blur fusion
    draw_arrow(ax, 7.8, 7.05, 7.8, 6.9, '#F57F17', lw=1.5)
    
    # ---- Center: Per-modality logits ----
    # Arrows from EEG and visual to logits
    draw_arrow(ax, 4.0, 4.95, 7.0, 4.95, EEG_EDGE, lw=2.5)
    draw_arrow(ax, 9.5, 6.2, 9.5, 5.45, VIS_EDGE, lw=2)
    
    draw_box(ax, 9.5, 5.1, 3.4, 0.55, 'Per-Modality Similarity\nLogits Matrix (200×200)', PROC_COLOR, PROC_EDGE, fontsize=9, fw='bold')
    ax.text(9.5, 4.7, 'eeg_embed @ vis_embed.T per modality', ha='center', fontsize=7, style='italic', color='#546E7A')
    
    # 9 modality labels
    draw_arrow(ax, 9.5, 4.55, 9.5, 4.0, PROC_EDGE, lw=2)
    
    # Seed averaging
    draw_box(ax, 9.5, 3.7, 3.4, 0.55, 'Seed Averaging\n(zscore → mean → zscore)', PROC_COLOR, PROC_EDGE, fontsize=9)
    
    draw_arrow(ax, 9.5, 3.4, 9.5, 2.85, PROC_EDGE, lw=2)
    
    # Unfold to show 9 modalities
    modalities = [
        ('edge\n0.243', '#1565C0'), ('msblur6\n0.223', '#1976D2'),
        ('vae\n0.223', '#42A5F5'), ('rn50\n0.099', '#7986CB'),
        ('vitb32\n0.085', '#9FA8DA'), ('depth\n0.043', '#90CAF9'),
        ('image\n0.043', '#BBDEFB'), ('dinov2\n0.043', '#E3F2FD'),
    ]
    total_w = len(modalities) * 1.6
    start_x = 9.5 - total_w / 2 + 0.8
    for i, (name, color) in enumerate(modalities):
        x = start_x + i * 1.6
        fw = 8
        box_h = FancyBboxPatch((x - 0.65, 2.05), 1.3, 0.55,
                               boxstyle="round,pad=0.08", facecolor=color,
                               edgecolor='#333333', linewidth=0.8, alpha=0.85)
        ax.add_patch(box_h)
        ax.text(x, 2.32, name, ha='center', va='center', fontsize=6.5,
                fontweight='bold', color='white' if i < 5 else '#1A237E', zorder=3)
    
    # Weighted sum label
    ax.text(9.5, 1.7, 'Weighted Sum', ha='center', fontsize=9, fontweight='bold', color='#2E7D32')
    draw_arrow(ax, 9.5, 2.05, 9.5, 1.85, ENS_EDGE, lw=2.5)
    
    # Final ensemble
    draw_box(ax, 9.5, 1.3, 4.5, 0.65, 'Ensemble Logits (200×200)', ENS_COLOR, ENS_EDGE, fontsize=11, fw='bold')
    
    # Output arrows
    draw_arrow(ax, 7.5, 1.3, 5.5, 0.7, '#2E7D32', lw=2.5)
    draw_arrow(ax, 11.5, 1.3, 13.0, 0.7, '#C62828', lw=2.5)
    
    # Greedy retrieval
    draw_box(ax, 4.5, 0.4, 3.2, 0.55, 'Greedy Top-1: 67.0%\nGreedy Top-5: 89.0%', '#E8F5E9', '#2E7D32', fontsize=9.5, fw='bold')
    
    # Hungarian matching
    draw_box(ax, 13.0, 0.4, 3.2, 0.55, 'Hungarian Top-1: 96.5%\nIH-Top-5: 99.5%', '#FCE4EC', '#C62828', fontsize=9.5, fw='bold')
    
    # Legend boxes for evaluation paradigms
    ax.text(4.5, 0.0, 'Open Retrieval', ha='center', fontsize=8, style='italic', color='#2E7D32')
    ax.text(13.0, 0.0, 'Closed-set Only', ha='center', fontsize=8, style='italic', color='#C62828')
    
    fig.savefig(os.path.join(FIG_DIR, "method_pipeline.pdf"), bbox_inches='tight', dpi=200)
    fig.savefig(os.path.join(FIG_DIR, "method_pipeline.png"), bbox_inches='tight', dpi=200)
    plt.close(fig)
    print("[OK] F1 v2: method_pipeline")

# ==============================================================================
# F3: Redesigned Foveated Blur Examples (with better visual presentation)
# ==============================================================================
def generate_foveated_blur_v2():
    sorted_ids = sorted(test_idx_to_id.keys())[:4]
    
    # Pick 4 visually interesting test images
    # Use indices that have distinct subjects
    chosen_indices = []
    for idx in sorted_ids:
        img_id = test_idx_to_id[idx]
        img_path = test_id_to_path[img_id]
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        if w >= 200 and h >= 200 and len(chosen_indices) < 4:
            chosen_indices.append(idx)
        if len(chosen_indices) >= 4:
            break
    if len(chosen_indices) < 4:
        chosen_indices = sorted_ids[:4]
    
    sigmas = [0, 2.0, 8.0, 14.0]
    blur_labels = ['Clean\n(σ=0)', 'Fovea-low\n(σ=2)', 'Fovea-mid\n(σ=8)', 'Fovea-high\n(σ=14)']
    
    n_rows = len(chosen_indices)
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4.2 * n_rows))
    
    img_size = 256
    
    for row, idx in enumerate(chosen_indices):
        img_id = test_idx_to_id[idx]
        img_path = test_id_to_path[img_id]
        img = Image.open(img_path).convert('RGB').resize((img_size, img_size), Image.LANCZOS)
        img_arr = np.asarray(img, dtype=np.float32)
        
        for col, (sigma, label) in enumerate(zip(sigmas, blur_labels)):
            if sigma == 0:
                result = img_arr.copy()
            else:
                blurred = img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))
                blurred_arr = np.asarray(blurred, dtype=np.float32)
                w, h = img_size, img_size
                yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
                cx, cy = w / 2, h / 2
                dist = np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
                mask = np.clip((dist / 0.55) ** 1.7, 0, 1)
                result = img_arr * (1 - mask[..., None]) + blurred_arr * mask[..., None]
            
            axes[row, col].imshow(result.astype(np.uint8))
            axes[row, col].axis('off')
            
            if row == 0:
                axes[row, col].set_title(label, fontsize=11, fontweight='bold', pad=10)
    
    # Add a text annotation explaining the effect
    fig.text(0.5, -0.02, 
             'Multi-scale foveated blur: central region stays sharp while periphery degrades with increasing σ.\n'
             'These 4 scales (clean, low, mid, high) are concatenated and projected to form the msblur6 visual feature.',
             ha='center', fontsize=10, style='italic', color='#555555',
             wrap=True)
    
    # Add grid lines
    for row in range(n_rows):
        for col in range(4):
            for spine in axes[row, col].spines.values():
                spine.set_visible(True)
                spine.set_color('#DDDDDD')
                spine.set_linewidth(0.5)
    
    fig.suptitle('Multi-Scale Foveated Blur: Central Acuity with Peripheral Degradation',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "foveated_blur_examples.pdf"), bbox_inches='tight', dpi=200)
    fig.savefig(os.path.join(FIG_DIR, "foveated_blur_examples.png"), bbox_inches='tight', dpi=200)
    plt.close(fig)
    print("[OK] F3 v2: foveated_blur_examples")


if __name__ == '__main__':
    generate_pipeline_v2()
    generate_foveated_blur_v2()
    
    # Copy to report package
    import shutil
    pkg_dir = os.path.join(ROOT, "report_figures_tables")
    os.makedirs(pkg_dir, exist_ok=True)
    
    for ext in ['pdf', 'png']:
        for name in ['method_pipeline', 'foveated_blur_examples']:
            src = os.path.join(FIG_DIR, f"{name}.{ext}")
            dst = os.path.join(pkg_dir, f"{name}.{ext}")
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"  Copied {name}.{ext} to report package")
    
    # Re-zip
    import zipfile
    zip_path = os.path.join(ROOT, "report_figures_tables.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in os.listdir(pkg_dir):
            fp = os.path.join(pkg_dir, f)
            if os.path.isfile(fp):
                zf.write(fp, os.path.join("report_figures_tables", f))
    print(f"\nRe-packaged: {zip_path}")