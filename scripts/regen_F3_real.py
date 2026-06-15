#!/usr/bin/env python3
"""Regenerate F3 with real test images showing foveated blur progressive effect."""

import os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageFilter

ROOT = "/hpc2hdd/JH_DATA/jhai_data/dsaa2012_031/project_codex"
FIG_DIR = os.path.join(ROOT, "figures")
TEST_DIR = os.path.join(ROOT, "image-eeg-data", "test_images")

# Choose 4 visually distinct, recognizable categories
chosen = [
    "00002_antelope",       # animal
    "00008_basketball",     # sports equipment
    "00015_birthday_cake",  # food
    "00001_aircraft_carrier",  # vehicle
]

def load_img(dir_name):
    d = os.path.join(TEST_DIR, dir_name)
    imgs = sorted([f for f in os.listdir(d) if f.endswith(('.jpg', '.png'))])
    if imgs:
        return Image.open(os.path.join(d, imgs[0])).convert('RGB')
    return None

def foveated_blur(img, sigma):
    if sigma <= 0:
        return np.asarray(img, dtype=np.float32)
    blurred = img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))
    w, h = img.size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt(((xx - h / 2) / w) ** 2 + ((yy - w / 2) / h) ** 2)
    mask = np.clip((dist / 0.55) ** 1.7, 0, 1)
    a = np.asarray(img, dtype=np.float32)
    b = np.asarray(blurred, dtype=np.float32)
    result = a * (1 - mask[..., None]) + b * mask[..., None]
    return result

sigmas = [0, 2.0, 8.0, 14.0]
col_titles = [
    'Clean (σ = 0)',
    'Fovea-low (σ = 2)',
    'Fovea-mid (σ = 8)',
    'Fovea-high (σ = 14)',
]
# Row labels derived from directory name
row_labels = [d.split('_', 1)[1].replace('_', ' ').title() for d in chosen]

n_rows = len(chosen)
n_cols = 4
img_display_size = 256

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.8 * n_rows + 1.2))

for row, dir_name in enumerate(chosen):
    img = load_img(dir_name)
    if img is None:
        continue
    img = img.resize((img_display_size, img_display_size), Image.LANCZOS)
    
    for col, (sigma, title) in enumerate(zip(sigmas, col_titles)):
        result = foveated_blur(img, sigma)
        axes[row, col].imshow(result.astype(np.uint8))
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        
        # Subtle border
        for spine in axes[row, col].spines.values():
            spine.set_visible(True)
            spine.set_color('#BDBDBD')
            spine.set_linewidth(0.8)

    # Row label on the left side
    axes[row, 0].set_ylabel(row_labels[row], fontsize=12, fontweight='bold',
                             rotation=0, labelpad=70, va='center')

# Column titles
for col, title in enumerate(col_titles):
    axes[0, col].set_title(title, fontsize=12, fontweight='bold', pad=12)

fig.suptitle('Multi-Scale Foveated Blur: Sharp Center with Progressive Peripheral Degradation',
             fontsize=15, fontweight='bold', y=1.01)

# Bottom annotation
fig.text(0.5, -0.02,
         'Each row shows the same test stimulus with increasing foveated blur. '
         'Central regions remain sharp while the periphery progressively degrades, '
         'simulating human retinal foveation.\n'
         'The 4 scales are concatenated as visual features for the msblur6 modality '
         '(edge: 24.3%, msblur6: 22.3% of ensemble weight).',
         ha='center', fontsize=10, style='italic', color='#555555', wrap=True)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "foveated_blur_examples.pdf"), bbox_inches='tight', dpi=200)
fig.savefig(os.path.join(FIG_DIR, "foveated_blur_examples.png"), bbox_inches='tight', dpi=200)
plt.close(fig)

# Copy to report package
import shutil
pkg_dir = os.path.join(ROOT, "report_figures_tables")
os.makedirs(pkg_dir, exist_ok=True)
for ext in ['pdf', 'png']:
    src = os.path.join(FIG_DIR, f"foveated_blur_examples.{ext}")
    dst = os.path.join(pkg_dir, f"foveated_blur_examples.{ext}")
    shutil.copy2(src, dst)
    print(f"Copied foveated_blur_examples.{ext} to report package")

# Re-zip
import zipfile
zip_path = os.path.join(ROOT, "report_figures_tables.zip")
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in os.listdir(pkg_dir):
        fp = os.path.join(pkg_dir, f)
        if os.path.isfile(fp):
            zf.write(fp, os.path.join("report_figures_tables", f))
print(f"Re-packaged: {zip_path}")
print("[OK] F3 v3: foveated_blur_examples with real test images")