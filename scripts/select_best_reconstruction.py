#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from eeg_cogcappro.utils import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select best legal reconstruction by fixed official-metric priority.")
    p.add_argument("--experiments-root", default="recons/experiments")
    p.add_argument("--results-root", default="results/reconstruction_experiments")
    p.add_argument("--output-dir", default="recons/atms_multimodal_final_improved")
    p.add_argument("--summary", default="results/reconstruction_experiments_summary.json")
    return p.parse_args()


def metric_key(item: dict) -> tuple[float, float, float]:
    metrics = item.get("metrics") or {}
    return (
        float(metrics.get("eval_clip", float("-inf"))),
        float(metrics.get("eval_alex5", float("-inf"))),
        float(metrics.get("eval_ssim", float("-inf"))),
    )


def main() -> None:
    args = parse_args()
    exp_root = Path(args.experiments_root)
    results_root = Path(args.results_root)
    items = []
    for exp_dir in sorted(p for p in exp_root.iterdir() if p.is_dir()):
        method = exp_dir.name
        pngs = sorted(exp_dir.glob("*.png"))
        summary_path = exp_dir / "summary.json"
        eval_path = results_root / f"{method}.json"
        method_summary = json.load(summary_path.open()) if summary_path.exists() else {}
        metrics = json.load(eval_path.open()) if eval_path.exists() else {}
        item = {
            "method": method,
            "reconstruction_dir": str(exp_dir),
            "num_png": len(pngs),
            "summary": method_summary,
            "metrics": metrics,
            "eligible": len(pngs) == 200 and method_summary.get("status") != "skipped",
            "selection_key": list(metric_key({"metrics": metrics})),
        }
        items.append(item)

    eligible = [item for item in items if item["eligible"]]
    if not eligible:
        raise RuntimeError("No eligible reconstruction experiment produced 200 PNGs.")
    best = max(eligible, key=metric_key)

    out_dir = Path(args.output_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    best_dir = Path(best["reconstruction_dir"])
    for path in sorted(best_dir.glob("*.png")):
        shutil.copy2(path, out_dir / path.name)
    for name in ["manifest.csv", "summary.json"]:
        src = best_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    summary = {
        "selection_rule": "max eval_clip, then eval_alex5, then eval_ssim",
        "best_method": best["method"],
        "best_reconstruction_dir": str(out_dir),
        "experiments": items,
    }
    write_json(args.summary, summary)
    print(f"Selected best reconstruction: {best['method']} -> {out_dir}")
    print(json.dumps(metric_key(best), indent=2))


if __name__ == "__main__":
    main()
