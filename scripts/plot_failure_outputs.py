#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
TOP5_CSV = ROOT / "results/multi_encoder_ensemble/retrieval_test_top5.csv"
RECON_SUMMARY = ROOT / "results/reconstruction_experiments_summary.json"
TEST_IMAGE_DIR = ROOT / "image-eeg-data/test_images"
RECON_DIR = ROOT / "recons/atms_multimodal_final_improved"
FIGURE_DIR = ROOT / "figures"


def read_top5_rows() -> list[dict[str, str]]:
    with TOP5_CSV.open(newline="") as f:
        return list(csv.DictReader(f))


def rank_of_gt(row: dict[str, str]) -> int | None:
    gt = row["image_id"]
    for rank in range(1, 6):
        if row[f"pred{rank}"] == gt:
            return rank
    return None


def concept_from_image_id(image_id: str) -> str:
    parts = image_id.split("_")
    if len(parts) <= 1:
        return image_id
    return "_".join(parts[:-1])


def build_image_lookup() -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for path in TEST_IMAGE_DIR.glob("*/*"):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            lookup[path.stem] = path
    return lookup


def open_square(path: Path, size: int = 224) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img = ImageOps.contain(img, (size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def plot_failure_statistics(rows: list[dict[str, str]]) -> Path:
    ranks = [rank_of_gt(row) for row in rows]
    rank_counts = [sum(r == k for r in ranks) for k in range(1, 6)]
    misses = sum(r is None for r in ranks)
    top1 = rank_counts[0]
    top2_5 = sum(rank_counts[1:])

    correct_margins = []
    wrong_margins = []
    for row, rank in zip(rows, ranks):
        margin = float(row["score1"]) - float(row["score2"])
        if rank == 1:
            correct_margins.append(margin)
        else:
            wrong_margins.append(margin)

    with RECON_SUMMARY.open() as f:
        recon_obj = json.load(f)
    experiments = recon_obj["experiments"]
    method_names = [e["method"] for e in experiments]
    clip_scores = [e["metrics"]["eval_clip"] for e in experiments]
    ssim_scores = [e["metrics"]["eval_ssim"] for e in experiments]
    alex5_scores = [e["metrics"]["eval_alex5"] for e in experiments]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.dpi": 150,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.suptitle("Failure Output Analysis: Retrieval and Reconstruction", fontsize=16, y=0.98)

    ax = axes[0, 0]
    categories = ["Top-1\ncorrect", "Rank 2-5\nnear miss", "Not in\nTop-5"]
    values = [top1, top2_5, misses]
    colors = ["#2E7D32", "#F9A825", "#C62828"]
    bars = ax.bar(categories, values, color=colors, edgecolor="#333333", linewidth=0.8)
    ax.set_ylabel("Number of test queries")
    ax.set_title("Retrieval outcome breakdown")
    ax.set_ylim(0, max(values) * 1.25)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 2, f"{val}\n{val / len(rows):.1%}", ha="center", va="bottom")

    ax = axes[0, 1]
    labels = ["Rank 1", "Rank 2", "Rank 3", "Rank 4", "Rank 5", "Miss"]
    counts = rank_counts + [misses]
    ax.bar(labels, counts, color=["#2E7D32"] + ["#F9A825"] * 4 + ["#C62828"], edgecolor="#333333", linewidth=0.8)
    ax.set_ylabel("Number of test queries")
    ax.set_title("Ground-truth rank distribution")
    ax.tick_params(axis="x", rotation=25)
    for i, val in enumerate(counts):
        ax.text(i, val + 1.5, str(val), ha="center", va="bottom")

    ax = axes[1, 0]
    box = ax.boxplot(
        [correct_margins, wrong_margins],
        labels=["Top-1 correct", "Top-1 wrong"],
        patch_artist=True,
        widths=0.55,
        showfliers=True,
    )
    for patch, color in zip(box["boxes"], ["#A5D6A7", "#EF9A9A"]):
        patch.set_facecolor(color)
    ax.set_ylabel("score1 - score2")
    ax.set_title("Confidence margin: correct vs failure")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.text(1, np.median(correct_margins), f"median={np.median(correct_margins):.2f}", ha="center", va="bottom")
    ax.text(2, np.median(wrong_margins), f"median={np.median(wrong_margins):.2f}", ha="center", va="bottom")

    ax = axes[1, 1]
    scatter = ax.scatter(ssim_scores, clip_scores, s=np.array(alex5_scores) * 260, c=clip_scores, cmap="viridis", edgecolor="#222222")
    for name, x, y in zip(method_names, ssim_scores, clip_scores):
        short = (
            name.replace("atms_ensemble_", "atms_")
            .replace("diffusion_", "diff_")
            .replace("train_nearest_", "tn_")
            .replace("concept_train_nearest", "concept_tn")
        )
        ax.annotate(short, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.set_xlabel("SSIM (pixel/structure fidelity)")
    ax.set_ylabel("CLIP score (semantic fidelity)")
    ax.set_title("Reconstruction trade-off: semantic vs pixel fidelity")
    ax.grid(True, linestyle="--", alpha=0.3)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("CLIP")

    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIGURE_DIR.mkdir(exist_ok=True)
    out = FIGURE_DIR / "failure_output_analysis.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def select_failure_cases(rows: list[dict[str, str]], n: int = 6) -> list[dict[str, str]]:
    enriched = []
    for row in rows:
        rank = rank_of_gt(row)
        if rank == 1:
            continue
        margin = float(row["score1"]) - float(row["score2"])
        row = dict(row)
        row["_rank"] = str(rank) if rank is not None else "miss"
        row["_margin"] = margin
        enriched.append(row)

    miss_cases = [r for r in enriched if r["_rank"] == "miss"]
    near_cases = [r for r in enriched if r["_rank"] != "miss"]
    miss_cases.sort(key=lambda r: r["_margin"], reverse=True)
    near_cases.sort(key=lambda r: r["_margin"], reverse=True)
    selected = miss_cases[: max(1, n // 2)] + near_cases[: n - max(1, n // 2)]
    return selected[:n]


def plot_failure_case_grid(rows: list[dict[str, str]]) -> Path:
    lookup = build_image_lookup()
    cases = select_failure_cases(rows, n=6)

    fig, axes = plt.subplots(len(cases), 3, figsize=(9.5, 2.65 * len(cases)))
    fig.suptitle("Qualitative Failure Outputs (evaluation-only GT/candidate views)", fontsize=15, y=0.995)
    if len(cases) == 1:
        axes = np.expand_dims(axes, 0)

    for r, row in enumerate(cases):
        idx = int(row["index"])
        gt_id = row["image_id"]
        pred_id = row["pred1"]
        recon_path = RECON_DIR / f"{idx:03d}.png"
        gt_path = lookup.get(gt_id)
        pred_path = lookup.get(pred_id)
        imgs = [
            open_square(gt_path) if gt_path else Image.new("RGB", (224, 224), "#eeeeee"),
            open_square(pred_path) if pred_path else Image.new("RGB", (224, 224), "#eeeeee"),
            open_square(recon_path) if recon_path.exists() else Image.new("RGB", (224, 224), "#eeeeee"),
        ]
        titles = [
            f"GT: {concept_from_image_id(gt_id)}",
            f"Top-1 pred: {concept_from_image_id(pred_id)}",
            f"Output #{idx:03d}",
        ]
        for c in range(3):
            axes[r, c].imshow(imgs[c])
            axes[r, c].set_title(titles[c], fontsize=10)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        rank_label = row["_rank"]
        subtitle = f"GT rank: {rank_label}; margin={row['_margin']:.2f}; top5={row['pred1']}, {row['pred2']}, {row['pred3']}, {row['pred4']}, {row['pred5']}"
        axes[r, 1].set_xlabel(subtitle, fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.985))
    FIGURE_DIR.mkdir(exist_ok=True)
    out = FIGURE_DIR / "failure_case_grid.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    rows = read_top5_rows()
    stat_path = plot_failure_statistics(rows)
    grid_path = plot_failure_case_grid(rows)
    print(stat_path)
    print(grid_path)


if __name__ == "__main__":
    main()
