#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import filecmp
import json
import math
import zipfile
from pathlib import Path
from typing import Any


FINAL_RETRIEVAL_DIR = Path("results/multi_encoder_ensemble")
FINAL_RETRIEVAL_LOGITS = FINAL_RETRIEVAL_DIR / "retrieval_test_logits.pt"
EXPECTED_WEIGHTS = {
    "image": 0.35,
    "depth": 0.15,
    "edge": 0.15,
    "rn50": 0.10,
    "vitb32": 0.10,
    "dinov2": 0.10,
    "vae": 0.05,
}
MIN_SOURCE_COUNTS = {
    "image": 10,
    "depth": 10,
    "edge": 10,
    "rn50": 3,
    "vitb32": 3,
    "dinov2": 3,
    "vae": 3,
}
SCRIPT_PATH_CHECKS = {
    Path("scripts/reconstruct_atms_final.sh"): [
        "results/multi_encoder_ensemble/retrieval_test_logits.pt",
    ],
    Path("scripts/run_reconstruction_experiments.sh"): [
        "results/multi_encoder_ensemble/retrieval_test_logits.pt",
    ],
    Path("scripts/package_final_submission.sh"): [
        "--retrieval-dir results/multi_encoder_ensemble",
    ],
    Path("scripts/package_improved_submission.sh"): [
        "--retrieval-dir results/multi_encoder_ensemble",
    ],
    Path("eeg_cogcappro/reconstruct_experiments.py"): [
        'default="results/multi_encoder_ensemble/retrieval_test_logits.pt"',
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate final retrieval/reconstruction submission artifacts.")
    p.add_argument("--retrieval-dir", default=str(FINAL_RETRIEVAL_DIR))
    p.add_argument("--retrieval-metrics", default=None)
    p.add_argument("--retrieval-logits", default=None)
    p.add_argument("--retrieval-top5", default=None)
    p.add_argument("--recon-dir", default="recons/atms_multimodal_final_improved")
    p.add_argument("--output-dir", default="outputs/atms_multimodal_final_improved")
    p.add_argument("--zip-path", default="outputs/atms_multimodal_final_improved/submission.zip")
    p.add_argument("--expected-recons", type=int, default=200)
    p.add_argument("--min-top1", type=float, default=0.475)
    p.add_argument("--min-top5", type=float, default=0.780)
    p.add_argument("--skip-candidate-order-check", action="store_true")
    p.add_argument("--skip-reconstruction-check", action="store_true")
    p.add_argument("--skip-script-path-check", action="store_true")
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _metrics_dict(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise KeyError(f"{path} lacks a metrics object")
    for key in ("top1_acc", "top5_acc"):
        if key not in metrics:
            raise KeyError(f"{path} metrics lacks {key}")
    return metrics


def _validate_metrics(metrics_path: Path, min_top1: float, min_top5: float) -> tuple[dict[str, Any], float, float]:
    payload = _load_json(metrics_path)
    metrics = _metrics_dict(payload, metrics_path)
    top1 = float(metrics["top1_acc"])
    top5 = float(metrics["top5_acc"])
    if not (math.isfinite(top1) and math.isfinite(top5)):
        raise RuntimeError(f"Non-finite retrieval metrics: top1={top1}, top5={top5}")
    if top1 <= min_top1:
        raise RuntimeError(f"Final Top-1 {top1:.4f} does not improve over threshold {min_top1:.4f}")
    if top5 <= min_top5:
        raise RuntimeError(f"Final Top-5 {top5:.4f} does not improve over threshold {min_top5:.4f}")
    return payload, top1, top5


def _validate_sources(metrics_payload: dict[str, Any]) -> list[Path]:
    weights = metrics_payload.get("weights")
    sources = metrics_payload.get("sources")
    if not isinstance(weights, dict):
        raise KeyError("Final metrics JSON lacks weights")
    if not isinstance(sources, dict):
        raise KeyError("Final metrics JSON lacks sources")

    for name, expected in EXPECTED_WEIGHTS.items():
        actual = float(weights.get(name, float("nan")))
        if abs(actual - expected) > 1e-8:
            raise RuntimeError(f"Unexpected ensemble weight for {name}: {actual} != {expected}")

    paths: list[Path] = []
    for name, min_count in MIN_SOURCE_COUNTS.items():
        modality_sources = sources.get(name)
        if not isinstance(modality_sources, list):
            raise RuntimeError(f"Missing source list for modality {name}")
        if len(modality_sources) < min_count:
            raise RuntimeError(f"Expected at least {min_count} {name} logits, found {len(modality_sources)}")
        for item in modality_sources:
            path = Path(item)
            if not path.exists():
                raise FileNotFoundError(path)
            paths.append(path)
    return paths


def _validate_candidate_order(final_logits_path: Path, source_paths: list[Path]) -> tuple[int, int]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "Candidate-order validation requires torch. Activate the project environment "
            "or pass --skip-candidate-order-check for packaging-only checks."
        ) from exc

    final_obj = torch.load(final_logits_path, map_location="cpu", weights_only=False)
    final_logits = final_obj.get("logits")
    if final_logits is None:
        raise KeyError(f"{final_logits_path} does not contain logits")
    if final_logits.ndim != 2:
        raise RuntimeError(f"{final_logits_path} logits must be 2D, got shape {tuple(final_logits.shape)}")
    reference_ids = final_obj.get("image_ids")

    for path in source_paths:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        logits = obj.get("logits")
        if logits is None:
            raise KeyError(f"{path} does not contain logits")
        if tuple(logits.shape) != tuple(final_logits.shape):
            raise RuntimeError(f"Logit shape mismatch in {path}: {tuple(logits.shape)} != {tuple(final_logits.shape)}")
        image_ids = obj.get("image_ids")
        if reference_ids is not None and image_ids is not None and list(image_ids) != list(reference_ids):
            raise RuntimeError(f"Candidate image order mismatch in {path}")
    return int(final_logits.shape[0]), int(final_logits.shape[1])


def _validate_script_paths() -> None:
    for path, needles in SCRIPT_PATH_CHECKS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8")
        missing = [needle for needle in needles if needle not in text]
        if missing:
            raise RuntimeError(f"{path} does not point to multi-encoder final artifacts: missing {missing}")


def _validate_reconstructions(recon_dir: Path, expected_recons: int) -> tuple[int, int]:
    manifest = recon_dir / "manifest.csv"
    pngs = sorted(recon_dir.glob("*.png"))
    if len(pngs) != expected_recons:
        raise RuntimeError(f"Expected {expected_recons} PNGs in {recon_dir}, found {len(pngs)}")
    if pngs[0].name != "000.png" or pngs[-1].name != f"{expected_recons - 1:03d}.png":
        raise RuntimeError(f"Unexpected PNG naming range: {pngs[0].name} .. {pngs[-1].name}")
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    bad = [row for row in rows if row.get("source") == "test_ground_truth" or row.get("source_kind") == "test_ground_truth"]
    if len(rows) != expected_recons:
        raise RuntimeError(f"Expected {expected_recons} manifest rows, found {len(rows)}")
    if bad:
        raise RuntimeError(f"Found {len(bad)} test_ground_truth manifest rows")
    return len(pngs), len(rows)


def _validate_packaged_outputs(
    output_dir: Path,
    zip_path: Path,
    retrieval_metrics_path: Path,
    retrieval_logits_path: Path,
    retrieval_top5_path: Path,
    expected_recons: int,
) -> int:
    for src, name in [
        (retrieval_metrics_path, "retrieval_test_metrics.json"),
        (retrieval_logits_path, "retrieval_test_logits.pt"),
        (retrieval_top5_path, "retrieval_test_top5.csv"),
    ]:
        staged = output_dir / name
        if not staged.exists():
            raise FileNotFoundError(staged)
        if not filecmp.cmp(src, staged, shallow=False):
            raise RuntimeError(f"Packaged {name} does not match {src}")

    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    zip_pngs = [n for n in names if n.startswith("reconstructions/") and n.endswith(".png")]
    required = {
        "retrieval_test_logits.pt",
        "retrieval_test_metrics.json",
        "retrieval_test_top5.csv",
        "reconstruction_manifest.csv",
        "reconstruction_summary.json",
    }
    missing = sorted(required.difference(names))
    if len(zip_pngs) != expected_recons or missing:
        raise RuntimeError(f"Bad zip contents: {len(zip_pngs)} PNGs, missing={missing}")
    return len(zip_pngs)


def main() -> None:
    args = parse_args()
    retrieval_dir = Path(args.retrieval_dir)
    metrics_path = Path(args.retrieval_metrics) if args.retrieval_metrics else retrieval_dir / "retrieval_test_metrics.json"
    logits_path = Path(args.retrieval_logits) if args.retrieval_logits else retrieval_dir / "retrieval_test_logits.pt"
    top5_path = Path(args.retrieval_top5) if args.retrieval_top5 else retrieval_dir / "retrieval_test_top5.csv"
    recon_dir = Path(args.recon_dir)
    output_dir = Path(args.output_dir)
    zip_path = Path(args.zip_path)

    for path in (metrics_path, logits_path, top5_path):
        if not path.exists():
            raise FileNotFoundError(path)

    metrics_payload, top1, top5 = _validate_metrics(metrics_path, args.min_top1, args.min_top5)
    source_paths = _validate_sources(metrics_payload)
    logit_rows = logit_cols = None
    if not args.skip_candidate_order_check:
        logit_rows, logit_cols = _validate_candidate_order(logits_path, source_paths)
    if not args.skip_script_path_check:
        _validate_script_paths()

    reconstruction_pngs = manifest_rows = zip_pngs = None
    if not args.skip_reconstruction_check:
        reconstruction_pngs, manifest_rows = _validate_reconstructions(recon_dir, args.expected_recons)
        zip_pngs = _validate_packaged_outputs(
            output_dir,
            zip_path,
            metrics_path,
            logits_path,
            top5_path,
            args.expected_recons,
        )

    print(
        {
            "retrieval_dir": str(retrieval_dir),
            "top1_acc": top1,
            "top5_acc": top5,
            "source_logits": len(source_paths),
            "logit_shape": [logit_rows, logit_cols] if logit_rows is not None else None,
            "reconstruction_pngs": reconstruction_pngs,
            "zip_pngs": zip_pngs,
            "manifest_rows": manifest_rows,
        }
    )


if __name__ == "__main__":
    main()
