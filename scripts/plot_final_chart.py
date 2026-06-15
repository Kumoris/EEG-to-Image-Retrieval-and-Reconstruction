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
    "Deep ViT-L edge",
    "ViT-B/32 (std)",
    "Deep ViT-B/32",
    "Deep ViT-L depth",
    "RN50 (std)",
    "Deep RN50",
    "Deep ViT-L image",
    "",
    "Ensemble\n(7-expert greedy)",
    "Deep ViT-L Ensemble\n(greedy)",
    "Deep ViT-L Ensemble\n+ Hungarian (Top-1)",
]

top1 = [10.17, 10.50, 12.33, 19.67, 23.85, 24.83, 24.17, 26.75, 26.00, 27.67, 27.45, 0, 50.00, 63.50, 94.50]
top5 = [30.83, 29.17, 34.83, 44.67, 56.50, 53.83, 56.17, 58.30, 56.67, 58.67, 58.45, 0, 85.00, 92.50, None]

x = np.arange(len(categories))
width = 0.38

fig, ax = plt.subplots(figsize=(18, 7))

c1 = "#4C72B0"
c2 = "#DD8452"
c_highlight1 = "#3A5BA0"
c_highlight2 = "#C44E52"

colors1 = [c1]*11 + ["white"] + [c_highlight1]*3
colors2 = [c2]*11 + ["white"] + [c_highlight2]*3

for i in range(len(categories)):
    if top1[i] == 0:
        continue
    ax.bar(x[i] - width/2, top1[i], width, color=colors1[i], edgecolor="white", linewidth=0.5)
    if top5[i] is not None:
        ax.bar(x[i] + width/2, top5[i], width, color=colors2[i], edgecolor="white", linewidth=0.5)

for i in range(len(categories)):
    if top1[i] == 0:
        continue
    if i >= 14:
        ax.text(x[i] - width/2, top1[i] + 1.0, f"{top1[i]:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax.text(x[i], top1[i] + 4, "Top-1 only\n(1-to-1 assign)", ha="center", va="bottom", fontsize=7, color="#C44E52", style="italic")
    elif i >= 12:
        ax.text(x[i] - width/2, top1[i] + 1.0, f"{top1[i]:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        if top5[i] is not None:
            ax.text(x[i] + width/2, top5[i] + 1.0, f"{top5[i]:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    else:
        ax.text(x[i] - width/2, top1[i] + 0.8, f"{top1[i]:.1f}", ha="center", va="bottom", fontsize=7)
        if top5[i] is not None:
            ax.text(x[i] + width/2, top5[i] + 0.8, f"{top5[i]:.1f}", ha="center", va="bottom", fontsize=7)

ax.bar(-1, 0, width, color=c1, label="Top-1 Accuracy (%)")
ax.bar(-1, 0, width, color=c2, label="Top-5 Accuracy (%)")

ax.set_ylabel("Accuracy (%)", fontsize=13)
ax.set_title("EEG-to-Image Retrieval: Experts → Ensemble → Hungarian Matching", fontsize=15, fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=8.5, rotation=35, ha="right")
ax.legend(fontsize=11, loc="upper left")
ax.set_ylim(0, 108)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.axvline(x=11.5, color="gray", linestyle=":", alpha=0.5, linewidth=1)
ax.axvline(x=13.5, color="gray", linestyle=":", alpha=0.5, linewidth=1)
ax.text(5.5, 103, "Individual Experts", ha="center", fontsize=10, color="gray", style="italic")
ax.text(12.5, 103, "Ensemble\n(greedy)", ha="center", fontsize=9, color="gray", style="italic")
ax.text(14.0, 103, "Hungarian\n(1-to-1)", ha="center", fontsize=9, color="gray", style="italic")

best_idx = 14
ax.annotate("",
            xy=(x[best_idx], 94.5), xytext=(x[best_idx], 100),
            arrowprops=dict(arrowstyle="-|>", color="#C44E52", lw=1.5),
            fontsize=9, color="#C44E52", ha="center")

plt.tight_layout()
out_path = "results/retrieval_final_chart.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved to {out_path}")
fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
print(f"Saved to {out_path.replace('.png', '.pdf')}")
