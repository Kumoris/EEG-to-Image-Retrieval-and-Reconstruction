#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


SEED_RE = re.compile(r"(.*?seed)\d+(.+)")


def _read_structured_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    if path.suffix.lower() == ".csv":
        with path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return {"rows": rows}
    if path.suffix.lower() in {".log", ".txt"}:
        return {"_text": path.read_text(encoding="utf-8", errors="replace")}
    raise ValueError(f"Unsupported result file type: {path}")


def _get_field(obj: dict[str, Any], field: str) -> float:
    if "_text" in obj:
        return _get_metric_from_text(str(obj["_text"]), field)

    cur: Any = obj
    for part in field.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise KeyError(f"Field '{field}' not found in result object.")
    if isinstance(cur, str):
        cur = cur.strip()
        try:
            return float(cur)
        except ValueError as exc:
            raise TypeError(f"Field '{field}' is not numeric: {cur!r}") from exc
    if not isinstance(cur, (int, float)):
        raise TypeError(f"Field '{field}' is not numeric: {cur!r}")
    return float(cur)


def _get_metric_from_text(text: str, field: str) -> float:
    names = [field]
    if "." in field:
        names.append(field.rsplit(".", 1)[-1])
    number = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    for name in names:
        pattern = re.compile(rf"(?<![\w.-]){re.escape(name)}(?![\w.-])\s*[:=]\s*{number}")
        match = pattern.search(text)
        if match:
            return float(match.group(1))
    raise KeyError(f"Metric '{field}' not found in text log.")


def _find_numeric_field(obj: dict[str, Any], candidates: list[str]) -> float:
    errors = []
    for field in candidates:
        try:
            return _get_field(obj, field)
        except Exception as exc:
            errors.append(str(exc))
    raise KeyError("None of the candidate fields were found: " + ", ".join(candidates) + "; " + " | ".join(errors[:3]))


def _seed_group_key(path: str | Path) -> str | None:
    match = SEED_RE.match(str(path))
    if not match:
        return None
    return f"{match.group(1)}*{match.group(2)}"


def _discover_seed_group(patterns: list[str], required_fields: list[list[str]], prefer: str | None = None) -> list[Path]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for pattern in patterns:
        for path in glob.glob(pattern):
            key = _seed_group_key(path)
            if key is not None:
                groups[key].append(Path(path))

    valid: list[tuple[str, list[Path]]] = []
    for key, paths in groups.items():
        ok_paths = []
        for path in sorted(paths):
            try:
                obj = _read_structured_file(path)
                for candidates in required_fields:
                    _find_numeric_field(obj, candidates)
                ok_paths.append(path)
            except Exception:
                continue
        if ok_paths:
            valid.append((key, ok_paths))

    if not valid:
        return []
    if prefer:
        preferred = [(k, p) for k, p in valid if prefer in k]
        if preferred:
            valid = preferred
    valid.sort(key=lambda item: (len(item[1]), item[0]), reverse=True)
    return valid[0][1]


def _expand_existing_files(pattern: str) -> list[Path]:
    paths = sorted(Path(p) for p in glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No result files matched pattern: {pattern}")
    return paths


def load_results(paths: list[Path], field_map: dict[str, list[str]]) -> dict[str, list[float]]:
    values = {name: [] for name in field_map}
    bad_files = []
    for path in paths:
        try:
            obj = _read_structured_file(path)
            records = obj["rows"] if isinstance(obj.get("rows"), list) else [obj]
            for record in records:
                row_values = {}
                for name, candidates in field_map.items():
                    row_values[name] = _find_numeric_field(record, candidates)
                for name, value in row_values.items():
                    values[name].append(value)
        except Exception as exc:
            bad_files.append(f"{path}: {exc}")
    if not any(values.values()):
        raise RuntimeError("Could not read any usable result values.\n" + "\n".join(bad_files[:10]))
    missing = [name for name, vals in values.items() if not vals]
    if missing:
        raise RuntimeError(f"Missing values for metrics: {missing}\n" + "\n".join(bad_files[:10]))
    return values


def aggregate_over_seeds(values: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for name, vals in values.items():
        arr = np.asarray(vals, dtype=np.float64)
        summary[name] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "n": int(len(arr)),
        }
    return summary


def _style_axes(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=18, pad=14)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, color="0.82", alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def _save_figure(fig, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")


def plot_retrieval_accuracy(
    summary: dict[str, dict[str, float]],
    output_base: Path,
    candidate_count: int = 200,
    title: str = "200-Way Retrieval Accuracy (mean ± std over seeds)",
) -> None:
    labels = ["Top-1", "Top-5"]
    means = np.array([summary["top1"]["mean"], summary["top5"]["mean"]]) * 100.0
    stds = np.array([summary["top1"]["std"], summary["top5"]["std"]]) * 100.0

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=6,
        width=0.58,
        color=["#4C78A8", "#72B7B2"],
        edgecolor="black",
        linewidth=0.8,
        error_kw={"elinewidth": 1.5, "capthick": 1.5},
    )
    top1_chance = 100.0 / candidate_count
    top5_chance = 100.0 * min(5, candidate_count) / candidate_count
    ax.axhline(top1_chance, color="#555555", linestyle="--", linewidth=1.3, label=f"Top-1 chance ({top1_chance:.1f}%)")
    ax.axhline(top5_chance, color="#999999", linestyle="--", linewidth=1.3, label=f"Top-5 chance ({top5_chance:.1f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(100.0, float((means + stds).max()) * 1.18))
    _style_axes(ax, title, "Accuracy (%)")
    ax.legend(frameon=False, fontsize=12, loc="upper left")

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(1.0, means.max() * 0.025),
            f"{mean:.1f}%",
            ha="center",
            va="bottom",
            fontsize=13,
        )
    plt.tight_layout()
    _save_figure(fig, output_base)
    plt.close(fig)


def plot_reconstruction_metrics(summary: dict[str, dict[str, float]], output_base: Path, metric_order: list[str]) -> None:
    labels = metric_order
    means = np.array([summary[name]["mean"] for name in labels], dtype=np.float64)
    stds = np.array([summary[name]["std"] for name in labels], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=6,
        width=0.62,
        color=["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"][: len(labels)],
        edgecolor="black",
        linewidth=0.8,
        error_kw={"elinewidth": 1.5, "capthick": 1.5},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    upper = float((means + stds).max())
    ax.set_ylim(0, min(1.05, upper * 1.18 if upper > 0 else 1.0))
    _style_axes(ax, "Reconstruction Metrics (mean ± std over seeds)", "Score")

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(0.015, upper * 0.025),
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )
    plt.tight_layout()
    _save_figure(fig, output_base)
    plt.close(fig)


def _parse_field_map(items: list[str], default: dict[str, list[str]]) -> dict[str, list[str]]:
    if not items:
        return default
    out: dict[str, list[str]] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected NAME=field1,field2 format, got: {item}")
        name, fields = item.split("=", 1)
        out[name] = [x.strip() for x in fields.split(",") if x.strip()]
    return out


def _auto_retrieval_paths(args: argparse.Namespace, field_map: dict[str, list[str]]) -> list[Path]:
    if args.retrieval_glob:
        return _expand_existing_files(args.retrieval_glob)
    paths = _discover_seed_group(
        ["results/*seed*_test*.json"],
        [field_map["top1"], field_map["top5"]],
        prefer=args.prefer_retrieval_group,
    )
    if not paths:
        raise FileNotFoundError(
            "Could not auto-discover retrieval seed JSON files. "
            "Pass --retrieval-glob, e.g. --retrieval-glob 'results/atms_vitl_seed*_test_tta0.json'."
        )
    return paths


def _load_final_retrieval_values(args: argparse.Namespace, field_map: dict[str, list[str]]) -> tuple[list[Path], dict[str, list[float]]]:
    path = Path(args.retrieval_final_json)
    if not path.exists():
        raise FileNotFoundError(
            f"Final retrieval summary not found: {path}. "
            "Pass --retrieval-source seeds or provide --retrieval-final-json."
        )

    obj = _read_structured_file(path)
    if "results" in obj and isinstance(obj["results"], dict):
        key = args.retrieval_final_key or obj.get("best")
        if not key or key not in obj["results"]:
            available = ", ".join(sorted(obj["results"])[:20])
            raise KeyError(f"Final retrieval key not found: {key!r}. Available keys include: {available}")
        obj = obj["results"][key]

    values = {}
    for name, candidates in field_map.items():
        values[name] = [_find_numeric_field(obj, candidates)]
    return [path], values


def _load_retrieval_values(args: argparse.Namespace, field_map: dict[str, list[str]]) -> tuple[list[Path], dict[str, list[float]], str]:
    use_final = args.retrieval_source == "final"
    if args.retrieval_source == "auto" and Path(args.retrieval_final_json).exists():
        use_final = True

    if use_final:
        paths, values = _load_final_retrieval_values(args, field_map)
        title = "200-Way Retrieval Accuracy (final multi-encoder ensemble)"
        return paths, values, title

    paths = _auto_retrieval_paths(args, field_map)
    values = load_results(paths, field_map)
    title = "200-Way Retrieval Accuracy (mean ± std over seeds)"
    return paths, values, title


def _auto_reconstruction_paths(args: argparse.Namespace, field_map: dict[str, list[str]]) -> list[Path]:
    if args.reconstruction_glob:
        return _expand_existing_files(args.reconstruction_glob)
    seed_paths = _discover_seed_group(["results/*recon*seed*.json", "results/*seed*recon*.json"], list(field_map.values()))
    if seed_paths:
        return seed_paths
    fallback = sorted(Path(p) for p in glob.glob("results/reconstruction_experiments/*.json"))
    if fallback:
        print(
            "Warning: no reconstruction seed-result files were found; "
            "using reconstruction experiment JSON files as repeated result files: results/reconstruction_experiments/*.json"
        )
        return fallback
    single = Path("results/atms_multimodal_final_improved_reconstruction_official.json")
    if single.exists():
        print("Warning: only one reconstruction metric JSON found; std will be 0.")
        return [single]
    raise FileNotFoundError(
        "Could not auto-discover reconstruction metric files. "
        "Pass --reconstruction-glob, e.g. --reconstruction-glob 'results/reconstruction_experiments/*.json'."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot paper-style retrieval and reconstruction bar charts from local result files.")
    p.add_argument(
        "--retrieval-source",
        choices=["auto", "final", "seeds"],
        default="auto",
        help="Use final ensemble summary if available, or seed files for mean/std over seeds.",
    )
    p.add_argument("--retrieval-final-json", default="results/multi_encoder_ensemble/retrieval_test_metrics.json")
    p.add_argument("--retrieval-final-key", default=None, help="Key under results[] in the final ensemble summary. Defaults to the JSON 'best' key.")
    p.add_argument("--retrieval-glob", default=None, help="Glob for retrieval seed JSON/CSV files. Auto-discovered if omitted.")
    p.add_argument("--reconstruction-glob", default=None, help="Glob for reconstruction metric JSON/CSV files. Auto-discovered if omitted.")
    p.add_argument("--prefer-retrieval-group", default="atms_vitl_seed", help="Substring preference when multiple retrieval seed groups exist.")
    p.add_argument("--output-dir", default="figures")
    p.add_argument("--candidate-count", type=int, default=200)
    p.add_argument("--retrieval-field", action="append", default=[], help="Override retrieval fields: name=field1,field2")
    p.add_argument("--reconstruction-field", action="append", default=[], help="Override reconstruction fields: name=field1,field2")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    retrieval_fields = _parse_field_map(
        args.retrieval_field,
        {
            "top1": ["top1_acc", "metrics.top1_acc"],
            "top5": ["top5_acc", "metrics.top5_acc"],
        },
    )
    reconstruction_fields = _parse_field_map(
        args.reconstruction_field,
        {
            "SSIM": ["eval_ssim", "metrics.eval_ssim"],
            "CLIP": ["eval_clip", "metrics.eval_clip"],
            "AlexNet-2": ["eval_alex2", "metrics.eval_alex2"],
            "AlexNet-5": ["eval_alex5", "metrics.eval_alex5"],
        },
    )

    retrieval_paths, retrieval_values, retrieval_title = _load_retrieval_values(args, retrieval_fields)
    reconstruction_paths = _auto_reconstruction_paths(args, reconstruction_fields)
    print(f"Retrieval source files ({len(retrieval_paths)}):")
    for path in retrieval_paths:
        print(f"  {path}")
    print(f"Reconstruction source files ({len(reconstruction_paths)}):")
    for path in reconstruction_paths:
        print(f"  {path}")

    reconstruction_values = load_results(reconstruction_paths, reconstruction_fields)
    retrieval_summary = aggregate_over_seeds(retrieval_values)
    reconstruction_summary = aggregate_over_seeds(reconstruction_values)

    output_dir = Path(args.output_dir)
    plot_retrieval_accuracy(
        retrieval_summary,
        output_dir / "retrieval_accuracy",
        candidate_count=args.candidate_count,
        title=retrieval_title,
    )
    plot_reconstruction_metrics(reconstruction_summary, output_dir / "reconstruction_metrics", list(reconstruction_fields.keys()))

    print("Retrieval summary:", retrieval_summary)
    print("Reconstruction summary:", reconstruction_summary)
    print(f"Saved figures to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
