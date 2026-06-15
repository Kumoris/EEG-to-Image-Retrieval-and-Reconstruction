#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eeg_cogcappro.utils import compute_retrieval_metrics, ensure_dir, write_csv, write_json


def _row_zscore(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    return (logits - logits.mean(dim=1, keepdim=True)) / logits.std(dim=1, keepdim=True).clamp_min(1e-6)


def _normalize(logits: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return logits.float()
    if mode == "row_zscore":
        return _row_zscore(logits)
    if mode == "softmax":
        return logits.float().softmax(dim=1)
    raise ValueError(f"Unknown normalize mode: {mode}")


def _parse_key_value(items: list[str] | None, value_type):
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        out[key.strip()] = value_type(value.strip())
    return out


def _load_logit_file(path: str | Path, normalize: str) -> tuple[torch.Tensor, dict]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if "logits" not in obj:
        raise KeyError(f"{path} does not contain a 'logits' tensor")
    return _normalize(obj["logits"], normalize), obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensemble retrieval logits with optional modality-level averaging.")
    parser.add_argument("--logits", action="append", default=[], help="Individual logits .pt file. May be repeated.")
    parser.add_argument(
        "--modality",
        action="append",
        default=[],
        help="Modality glob in NAME=GLOB form, e.g. image='results/atms_vitl_seed*_test_tta0.logits.pt'.",
    )
    parser.add_argument(
        "--weights",
        action="append",
        default=[],
        help="Modality weight in NAME=FLOAT form. If omitted, modalities/logits are equally weighted.",
    )
    parser.add_argument("--normalize", choices=["row_zscore", "softmax", "none"], default="row_zscore")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--hungarian", action="store_true", help="Apply Hungarian matching for 1-to-1 assignment (only for square logits).")
    parser.add_argument("--hungarian-topk", type=int, default=5, help="Number of iterative Hungarian rounds for Top-K evaluation (default: 5).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    modality_globs = _parse_key_value(args.modality, str)
    weights = _parse_key_value(args.weights, float)
    ensemble_terms: list[tuple[str, torch.Tensor, list[str]]] = []
    first_meta: dict | None = None
    image_ids = None
    concepts = None

    def check_meta(meta: dict, path: str) -> None:
        nonlocal first_meta, image_ids, concepts
        if first_meta is None:
            first_meta = meta
            image_ids = meta.get("image_ids")
            concepts = meta.get("concepts")
            return
        if image_ids is not None and meta.get("image_ids") is not None and meta.get("image_ids") != image_ids:
            raise ValueError(f"Candidate image order mismatch in {path}")

    for path in args.logits:
        logits, meta = _load_logit_file(path, args.normalize)
        check_meta(meta, path)
        ensemble_terms.append((str(path), logits, [str(path)]))

    for name, pattern in modality_globs.items():
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No logits matched modality {name}: {pattern}")
        tensors = []
        for path in paths:
            logits, meta = _load_logit_file(path, args.normalize)
            check_meta(meta, path)
            tensors.append(logits)
        modality_logits = _normalize(torch.stack(tensors).mean(dim=0), args.normalize)
        ensemble_terms.append((name, modality_logits, paths))

    if not ensemble_terms:
        raise ValueError("Provide at least one --logits or --modality entry.")

    if weights:
        missing = [name for name, _, _ in ensemble_terms if name not in weights]
        if missing:
            raise ValueError(f"Missing --weights for ensemble terms: {missing}")
        weight_sum = sum(weights[name] for name, _, _ in ensemble_terms)
        logits = sum(tensor * (weights[name] / weight_sum) for name, tensor, _ in ensemble_terms)
        used_weights = {name: weights[name] / weight_sum for name, _, _ in ensemble_terms}
    else:
        logits = torch.stack([tensor for _, tensor, _ in ensemble_terms]).mean(dim=0)
        used_weights = {name: 1.0 / len(ensemble_terms) for name, _, _ in ensemble_terms}

    metrics = compute_retrieval_metrics(logits)

    hungarian_metrics = None
    col_ind = None
    if args.hungarian and logits.shape[0] == logits.shape[1]:
        try:
            from scipy.optimize import linear_sum_assignment
            import numpy as np
            cost = -logits.float().numpy()
            row_ind, col_ind = linear_sum_assignment(cost)
            N = logits.shape[0]
            targets = np.arange(N)
            h_top1 = (col_ind == targets).mean()
            h_top1_count = int((col_ind == targets).sum())
            t_targets = torch.arange(N)
            greedy_top1_count = int((logits.argmax(dim=1) == t_targets).sum())

            mat = cost.copy()
            assignments = [(np.arange(N), col_ind)]
            iter_topk = {1: float(h_top1)}
            for k in range(2, args.hungarian_topk + 1):
                masked = mat.copy()
                for prev_rows, prev_cols in assignments:
                    masked[prev_rows, prev_cols] = 1e9
                _, col_k = linear_sum_assignment(masked)
                assignments.append((np.arange(N), col_k))
                all_cand = np.stack([a[1] for a in assignments], axis=1)
                hit = (all_cand == targets[:, None]).any(axis=1).mean()
                iter_topk[k] = float(hit)

            all_cand_5 = np.stack([a[1] for a in assignments[:min(len(assignments), 5)]], axis=1)
            h_top5_hung = float((all_cand_5 == targets[:, None]).any(axis=1).mean())

            hungarian_metrics = {
                "top1_acc": float(h_top1), "top1_count": h_top1_count,
                "top5_acc": h_top5_hung, "total": N,
                "greedy_top1_acc": float(greedy_top1_count / N),
                "greedy_top1_count": greedy_top1_count,
                "greedy_top5_acc": float(metrics["top5_acc"]),
                "net_gain": h_top1_count - greedy_top1_count,
                "iterative_hungarian_topk": iter_topk,
                "hungarian_topk": args.hungarian_topk,
                "metric_scope": {
                    "hungarian_top1": (
                        "Closed-set bipartite optimal assignment (Kuhn-Munkres algorithm). "
                        "Solves a global 1-to-1 matching over the N_query x N_candidate similarity matrix. "
                        "Requires N_query == N_candidate (square matrix). "
                        "This metric is NOT directly comparable to standard retrieval Top-1 "
                        "(which allows multiple queries to match the same candidate). "
                        "Applicable only to closed-set scenarios with equal query and gallery sizes."
                    ),
                    "iterative_hungarian_topk": (
                        "K-best bipartite matching via successive Hungarian rounds. "
                        "After each optimal assignment, matched cells are masked and the assignment "
                        "is repeated. For each query, the union of candidates across all rounds "
                        "forms the Top-K candidate set. Provides a principled closed-set Top-K "
                        "extension of Hungarian matching (Chegireddy & Hamacher, 1987). "
                        "NOT directly comparable to standard greedy retrieval Top-K."
                    ),
                    "greedy_top5": (
                        "Standard row-independent retrieval metric. "
                        "Each query independently selects its top-k candidates. "
                        "Directly comparable across methods and applicable to both open and closed retrieval."
                    ),
                },
            }
        except ImportError:
            print("Warning: scipy not available, skipping Hungarian matching", flush=True)

    ranks = logits.topk(k=min(args.topk, logits.shape[1]), dim=1)
    if image_ids is None:
        image_ids = [str(i) for i in range(logits.shape[0])]
    if concepts is None:
        concepts = [""] * logits.shape[0]

    rows = []
    for i, image_id in enumerate(image_ids):
        row = {"index": i, "image_id": image_id, "gt_concept": concepts[i] if i < len(concepts) else ""}
        for j in range(ranks.indices.shape[1]):
            cand = int(ranks.indices[i, j])
            row[f"pred{j+1}"] = image_ids[cand]
            row[f"score{j+1}"] = float(ranks.values[i, j])
        rows.append(row)
    fields = ["index", "image_id", "gt_concept"] + [x for j in range(1, ranks.indices.shape[1] + 1) for x in (f"pred{j}", f"score{j}")]

    summary = {
        "split": args.split,
        "normalize": args.normalize,
        "metric_scope": {
            "greedy_top1": "Standard retrieval: each query independently selects argmax candidate. Applicable to both open and closed retrieval.",
            "greedy_top5": "Standard retrieval: each query independently selects top-5 candidates. Applicable to both open and closed retrieval.",
            "hungarian_top1": (
                "Closed-set bipartite optimal assignment (Kuhn-Munkres). "
                "Requires N_query == N_candidate. NOT directly comparable to standard retrieval Top-1. "
                "Applicable only to closed-set scenarios with equal query and gallery sizes."
            ),
        },
        "metrics": metrics,
        "hungarian": hungarian_metrics,
        "weights": used_weights,
        "sources": {name: sources for name, _, sources in ensemble_terms},
    }
    write_json(output_dir / f"retrieval_{args.split}_metrics.json", summary)
    write_csv(output_dir / f"retrieval_{args.split}_top{ranks.indices.shape[1]}.csv", rows, fields)
    torch.save(
        {"logits": logits, "image_ids": image_ids, "concepts": concepts,
         "hungarian_assignment": col_ind.tolist() if hungarian_metrics else None,
         **summary},
        output_dir / f"retrieval_{args.split}_logits.pt",
    )
    print(f"Wrote ensemble retrieval outputs: {output_dir}")
    print(summary)
    if hungarian_metrics:
        print(f"Hungarian: top1={hungarian_metrics['top1_acc']*100:.1f}% "
              f"({hungarian_metrics['top1_count']}/{hungarian_metrics['total']}), "
              f"greedy was {hungarian_metrics['greedy_top1_count']}/{hungarian_metrics['total']}, "
              f"net gain={hungarian_metrics['net_gain']:+d}")
        iter_tk = hungarian_metrics.get("iterative_hungarian_topk", {})
        if iter_tk:
            parts = [f"Top-{k}={v*100:.1f}%" for k, v in sorted(iter_tk.items())]
            print(f"Iterative Hungarian: {', '.join(parts)}")
        print(f"[NOTE] Hungarian metrics use forced 1-to-1 bipartite assignment (Kuhn 1955; "
              f"Munkres 1957) and are NOT directly comparable to standard retrieval Top-1/Top-5.")


if __name__ == "__main__":
    main()
