#!/usr/bin/env python3
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

categories = [
    "VAE (std)",
    "Deep VAE",
    "DINOv2 (std)",
    "Deep DINOv2",
    "ViT-L edge (std)",
    "ViT-L depth (std)",
    "ViT-B/32 (std)",
    "Deep ViT-B/32",
    "RN50 (std)",
    "Deep RN50",
    "ViT-L image (std)",
    "",
    "Ensemble v1\n(7-expert)",
    "Ensemble v2\n(optimized)",
]

top1 = [10.17, 10.50, 12.33, 19.67, 20.00, 20.90, 24.83, 24.17, 26.00, 27.67, 28.20, 0, 50.00, 66.00]
top5 = [30.83, 29.17, 34.83, 44.67, 50.40, 51.90, 53.83, 56.17, 56.67, 58.67, 58.65, 0, 85.00, 90.00]

x = np.arange(len(categories))
width = 0.38

fig, ax = plt.subplots(figsize=(16, 7))

bars1 = ax.bar(x - width / 2, top1, width, label="Top-1 Accuracy (%)", color="#4C72B0", edgecolor="white", linewidth=0.5)
bars2 = ax.bar(x + width / 2, top5, width, label="Top-5 Accuracy (%)", color="#DD8452", edgecolor="white", linewidth=0.5)

for bar in bars1:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

for bar in bars2:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax.set_ylabel("Accuracy (%)", fontsize=13)
ax.set_title("EEG-to-Image Retrieval: Individual Experts vs Ensemble", fontsize=15, fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=9, rotation=30, ha="right")
ax.legend(fontsize=11, loc="upper left")
ax.set_ylim(0, 105)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.axvline(x=11.5, color="gray", linestyle=":", alpha=0.5, linewidth=1)
ax.text(5.5, 98, "Individual Experts", ha="center", fontsize=10, color="gray", style="italic")
ax.text(13.0, 98, "Ensemble", ha="center", fontsize=10, color="gray", style="italic")

plt.tight_layout()
out_path = "results/retrieval_comparison_chart.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved to {out_path}")
fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
print(f"Saved to {out_path.replace('.png', '.pdf')}")
