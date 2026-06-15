#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1] if "scripts" in str(__file__) else Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

MODALITY_GLOBS_TEST = {
    "image": "deep_vitl_image_seed*_test_tta5.logits.pt",
    "depth": "deep_vitl_depth_seed*_test_tta5.logits.pt",
    "edge": "deep_vitl_edge_seed*_test_tta5.logits.pt",
    "deep_rn50": "deep_rn50_seed*_test_tta5.logits.pt",
    "deep_vitb32": "deep_vit_b_32_seed*_test_tta5.logits.pt",
    "deep_dinov2": "deep_dinov2_da2_seed*_test_tta5.logits.pt",
    "deep_vae": "deep_vae_seed*_test_tta5.logits.pt",
    "msblur6": "deep_linear_seed*_test_tta5.logits.pt",
    "msblur2": "deep_multiscale_seed*_test_tta5.logits.pt",
}

MODALITY_GLOBS_TEST_FALLBACK = {
    "image": "atms_vitl_seed*_test_tta0.logits.pt",
    "depth": "atms_depth_vitl_seed*_test_tta0.logits.pt",
    "edge": "atms_edge_vitl_seed*_test_tta0.logits.pt",
    "deep_rn50": "deep_rn50_seed*_test_tta5.logits.pt",
    "deep_vitb32": "deep_vit_b_32_seed*_test_tta5.logits.pt",
    "deep_dinov2": "deep_dinov2_da2_seed*_test_tta5.logits.pt",
    "deep_vae": "deep_vae_seed*_test_tta5.logits.pt",
    "msblur6": "deep_linear_seed*_test_tta5.logits.pt",
    "msblur2": "deep_multiscale_seed*_test_tta5.logits.pt",
}

MODALITY_GLOBS_TRAIN = {
    "image": "deep_vitl_image_seed*_train_tta5.logits.pt",
    "depth": "deep_vitl_depth_seed*_train_tta5.logits.pt",
    "edge": "deep_vitl_edge_seed*_train_tta5.logits.pt",
    "deep_rn50": "deep_rn50_seed*_train_tta5.logits.pt",
    "deep_vitb32": "deep_vit_b_32_seed*_train_tta5.logits.pt",
    "deep_dinov2": "deep_dinov2_da2_seed*_train_tta5.logits.pt",
    "deep_vae": "deep_vae_seed*_train_tta5.logits.pt",
    "msblur6": "deep_linear_seed*_train_tta5.logits.pt",
    "msblur2": "deep_multiscale_seed*_train_tta5.logits.pt",
}

CURRENT_WEIGHTS = {
    "image": 0.35,
    "depth": 0.15,
    "edge": 0.15,
    "deep_rn50": 0.10,
    "deep_vitb32": 0.10,
    "deep_dinov2": 0.10,
    "deep_vae": 0.05,
}


def row_zscore(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    return (logits - logits.mean(dim=1, keepdim=True)) / logits.std(dim=1, keepdim=True).clamp(min=1e-6)


def load_modality_logits(results_dir: Path, glob_dict: dict[str, str],
                         fallback_dict: dict[str, str] | None = None) -> dict[str, torch.Tensor]:
    import glob as _glob
    if fallback_dict is None:
        fallback_dict = glob_dict
    out = {}
    for name, pattern in glob_dict.items():
        full_pattern = str(results_dir / pattern)
        paths = sorted(_glob.glob(full_pattern))
        if not paths:
            fallback_pattern = str(results_dir / fallback_dict.get(name, pattern))
            paths = sorted(_glob.glob(fallback_pattern))
        if not paths:
            print(f"  Warning: no logits for {name}", file=sys.stderr)
            continue
        source = "deep" if "deep_vitl" in paths[0] else "standard"
        tensors = [row_zscore(torch.load(p, map_location="cpu", weights_only=False)["logits"]) for p in paths]
        out[name] = row_zscore(torch.stack(tensors).mean(dim=0))
        print(f"  {name}: {len(paths)} files ({source})", file=sys.stderr)
    return out


def subsample_train_logits(train_logits: dict[str, torch.Tensor],
                           n_concepts: int = 200, seed: int = 0) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    n_total = next(iter(train_logits.values())).shape[0]
    idx = sorted(rng.choice(n_total, size=min(n_concepts, n_total), replace=False).tolist())
    out = {}
    for name, t in train_logits.items():
        sub = t[idx][:, idx]
        out[name] = row_zscore(sub)
    return out


def eval_batch(modality_stack: torch.Tensor, weight_matrix: np.ndarray,
               hungarian: bool = False) -> np.ndarray:
    W = torch.from_numpy(weight_matrix).float()
    W = W / W.sum(dim=1, keepdim=True).clamp(min=1e-8)
    combined = torch.einsum("bm,mdp->bdp", W, modality_stack)
    n_test = combined.shape[1]
    targets = torch.arange(n_test, device=combined.device)
    top1 = combined.argmax(dim=2)
    t1_greedy = (top1 == targets[None, :]).float().mean(dim=1)
    topk = combined.topk(k=min(5, n_test), dim=2).indices
    t5_greedy = (topk == targets[None, :, None]).any(dim=2).float().mean(dim=1)

    if hungarian and combined.shape[1] == combined.shape[2]:
        from scipy.optimize import linear_sum_assignment
        t1_hung = []
        for b in range(combined.shape[0]):
            _, col = linear_sum_assignment(-combined[b].numpy())
            t1_hung.append(float((col == np.arange(n_test)).mean()))
        t1_hung = torch.tensor(t1_hung)
        return torch.stack([t1_hung, t5_greedy], dim=1).numpy()

    return torch.stack([t1_greedy, t5_greedy], dim=1).numpy()


def random_search(modality_stack: torch.Tensor, names: list[str], n_trials: int = 200000,
                  top_n: int = 20, hungarian: bool = False) -> list[dict]:
    n = len(names)
    rng = np.random.default_rng(42)
    batch_size = 5000
    all_top1 = []
    all_top5 = []
    all_weights = []
    for start in range(0, n_trials, batch_size):
        bs = min(batch_size, n_trials - start)
        raw = rng.dirichlet(np.ones(n), size=bs)
        results = eval_batch(modality_stack, raw, hungarian=hungarian)
        all_top1.append(results[:, 0])
        all_top5.append(results[:, 1])
        all_weights.append(raw)
        if start + batch_size < n_trials and (start // batch_size) % 10 == 0:
            print(f"  random search: {start}/{n_trials}", file=sys.stderr)
    top1 = np.concatenate(all_top1)
    top5 = np.concatenate(all_top5)
    weights = np.concatenate(all_weights, axis=0)
    order = np.lexsort((-top5, -top1))
    results = []
    for i in order[:top_n]:
        results.append({
            "top1_acc": float(top1[i]),
            "top5_acc": float(top5[i]),
            "weights": {names[j]: float(weights[i, j]) for j in range(n)},
        })
    return results


def coordinate_descent(modality_stack: torch.Tensor, names: list[str],
                       init_w: dict[str, float], step: float = 0.01, rounds: int = 5,
                       hungarian: bool = False) -> dict:
    n = len(names)
    w = np.array([init_w.get(nm, 0.1) for nm in names], dtype=np.float64)
    w /= w.sum()

    def eval_w(wv):
        return eval_batch(modality_stack, wv.reshape(1, -1), hungarian=hungarian)[0]

    best = eval_w(w)
    print(f"  CD init: top1={best[0]:.4f} top5={best[1]:.4f}", file=sys.stderr)

    for rd in range(rounds):
        improved = False
        for dim in range(n):
            orig = w[dim]
            deltas = np.arange(-0.25, 0.26, step)
            candidates = []
            for d in deltas:
                w_test = w.copy()
                w_test[dim] = max(0.0, orig + d)
                s = w_test.sum()
                if s < 1e-8:
                    continue
                w_test /= s
                candidates.append(w_test)
            cmat = np.array(candidates)
            results = eval_batch(modality_stack, cmat, hungarian=hungarian)
            best_idx = np.lexsort((-results[:, 1], -results[:, 0]))[0]
            if results[best_idx, 0] > best[0] + 1e-6 or (
                abs(results[best_idx, 0] - best[0]) < 1e-6 and results[best_idx, 1] > best[1] + 1e-6
            ):
                best = results[best_idx]
                w = candidates[best_idx]
                improved = True
                print(f"  CD rd{rd} {names[dim]}: top1={best[0]:.4f} top5={best[1]:.4f}", file=sys.stderr)
        if not improved:
            print(f"  CD rd{rd}: converged", file=sys.stderr)
            break

    return {
        "top1_acc": float(best[0]),
        "top5_acc": float(best[1]),
        "weights": {names[i]: float(w[i]) for i in range(n)},
    }


def fine_grid(modality_stack: torch.Tensor, names: list[str], center: dict[str, float],
              radius: float = 0.10, step: float = 0.02, top_n: int = 20,
              hungarian: bool = False) -> list[dict]:
    n = len(names)
    centers = np.array([center.get(nm, 0.1) for nm in names])
    ranges = []
    for i in range(n):
        lo = max(0.0, centers[i] - radius)
        hi = min(1.0, centers[i] + radius)
        ranges.append(np.arange(lo, hi + step / 2, step))
    grids = np.meshgrid(*ranges, indexing="ij")
    all_w = np.stack([g.ravel() for g in grids], axis=1)
    row_sums = all_w.sum(axis=1)
    valid = row_sums > 1e-8
    all_w = all_w[valid]
    all_w = all_w / row_sums[valid, None]

    print(f"  fine grid: {len(all_w)} combos around center", file=sys.stderr)
    batch_size = 10000
    all_top1, all_top5 = [], []
    for start in range(0, len(all_w), batch_size):
        batch = all_w[start:start + batch_size]
        results = eval_batch(modality_stack, batch, hungarian=hungarian)
        all_top1.append(results[:, 0])
        all_top5.append(results[:, 1])
    top1 = np.concatenate(all_top1)
    top5 = np.concatenate(all_top5)
    order = np.lexsort((-top5, -top1))
    results = []
    for i in order[:top_n]:
        results.append({
            "top1_acc": float(top1[i]),
            "top5_acc": float(top5[i]),
            "weights": {names[j]: float(all_w[i, j]) for j in range(n)},
        })
    return results


def run_optimization(modality_stack: torch.Tensor, names: list[str],
                     hungarian: bool = False) -> list[dict]:
    mode_label = "Hungarian" if hungarian else "Greedy"

    cur_w = np.array([CURRENT_WEIGHTS.get(nm, 0.1) for nm in names], dtype=np.float64)
    cur_w /= cur_w.sum()
    cur_metrics = eval_batch(modality_stack, cur_w.reshape(1, -1), hungarian=hungarian)[0]
    print(f"\nCurrent ({mode_label}): top1={cur_metrics[0]:.4f} top5={cur_metrics[1]:.4f}", file=sys.stderr)

    print(f"\n=== Random search (200k trials, {mode_label}) ===", file=sys.stderr)
    rand_results = random_search(modality_stack, names, n_trials=200000, top_n=20, hungarian=hungarian)
    print(f"  Best random: top1={rand_results[0]['top1_acc']:.4f} top5={rand_results[0]['top5_acc']:.4f}", file=sys.stderr)

    print(f"\n=== Coordinate descent from current ({mode_label}) ===", file=sys.stderr)
    cd_current = coordinate_descent(modality_stack, names, CURRENT_WEIGHTS, step=0.01, rounds=5, hungarian=hungarian)
    print(f"  Result: top1={cd_current['top1_acc']:.4f} top5={cd_current['top5_acc']:.4f}", file=sys.stderr)

    print(f"\n=== Coordinate descent from best random ({mode_label}) ===", file=sys.stderr)
    cd_random = coordinate_descent(modality_stack, names, rand_results[0]["weights"], step=0.005, rounds=5, hungarian=hungarian)
    print(f"  Result: top1={cd_random['top1_acc']:.4f} top5={cd_random['top5_acc']:.4f}", file=sys.stderr)

    print(f"\n=== Fine grid around best CD result ({mode_label}) ===", file=sys.stderr)
    best_cd = cd_current if cd_current["top1_acc"] >= cd_random["top1_acc"] else cd_random
    fine_results = fine_grid(modality_stack, names, best_cd["weights"], radius=0.08, step=0.04, top_n=10, hungarian=hungarian)
    print(f"  Best fine grid: top1={fine_results[0]['top1_acc']:.4f} top5={fine_results[0]['top5_acc']:.4f}", file=sys.stderr)

    all_candidates = rand_results + [cd_current, cd_random] + fine_results
    all_candidates.sort(key=lambda x: (-x["top1_acc"], -x["top5_acc"]))
    return all_candidates


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--hungarian", action="store_true", help="Optimize for Hungarian matching Top-1 instead of greedy.")
    p.add_argument("--train-optimize", action="store_true",
                   help="Optimize weights on train logits, evaluate on test logits (honest evaluation).")
    p.add_argument("--n-subsamples", type=int, default=10,
                   help="Number of train subsampling iterations (default 10).")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir

    use_hung = args.hungarian

    if args.train_optimize:
        print("=" * 80, file=sys.stderr)
        print("MODE: Train-optimize + Test-evaluate (honest)", file=sys.stderr)
        print("=" * 80, file=sys.stderr)

        print("\nLoading TEST modality logits...", file=sys.stderr)
        test_logits = load_modality_logits(results_dir, MODALITY_GLOBS_TEST, MODALITY_GLOBS_TEST_FALLBACK)
        print(f"Loaded {len(test_logits)} test modalities", file=sys.stderr)

        print("\nLoading TRAIN modality logits...", file=sys.stderr)
        train_logits = load_modality_logits(results_dir, MODALITY_GLOBS_TRAIN)
        print(f"Loaded {len(train_logits)} train modalities", file=sys.stderr)

        common_names = sorted(set(test_logits.keys()) & set(train_logits.keys()))
        print(f"Common modalities: {common_names}", file=sys.stderr)

        test_stack = torch.stack([test_logits[nm] for nm in common_names], dim=0)
        print(f"Test stack shape: {test_stack.shape}", file=sys.stderr)

        n_train = next(iter(train_logits.values())).shape[0]
        print(f"Train logits full size: {n_train}x{n_train}", file=sys.stderr)

        best_overall = None
        best_avg_train_score = -1
        all_subsample_results = []

        for sub_idx in range(args.n_subsamples):
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"Subsample {sub_idx + 1}/{args.n_subsamples} (seed={sub_idx})", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)

            sub_train = subsample_train_logits(train_logits, n_concepts=200, seed=sub_idx)
            sub_stack = torch.stack([sub_train[nm] for nm in common_names], dim=0)
            print(f"Subsampled train stack shape: {sub_stack.shape}", file=sys.stderr)

            candidates = run_optimization(sub_stack, common_names, hungarian=use_hung)

            best_sub = candidates[0]
            avg_score = best_sub["top1_acc"]
            all_subsample_results.append({
                "subsample_idx": sub_idx,
                "train_top1": best_sub["top1_acc"],
                "train_top5": best_sub["top5_acc"],
                "weights": best_sub["weights"],
            })

            if avg_score > best_avg_train_score:
                best_avg_train_score = avg_score
                best_overall = best_sub

            print(f"\n  Subsample {sub_idx} best: train_top1={best_sub['top1_acc']:.4f} "
                  f"train_top5={best_sub['top5_acc']:.4f}", file=sys.stderr)

        print(f"\n{'='*80}", file=sys.stderr)
        print("Evaluating train-optimized weights on TEST set...", file=sys.stderr)
        print(f"{'='*80}", file=sys.stderr)

        w_arr = np.array([best_overall["weights"][nm] for nm in common_names], dtype=np.float64)
        w_arr /= w_arr.sum()
        test_metrics = eval_batch(test_stack, w_arr.reshape(1, -1), hungarian=use_hung)[0]

        print(f"\nTest results (train-optimized weights):", file=sys.stderr)
        print(f"  Greedy Top-1: {test_metrics[0]*100:.1f}%", file=sys.stderr)
        print(f"  Greedy Top-5: {test_metrics[1]*100:.1f}%", file=sys.stderr)

        if use_hung:
            test_metrics_greedy = eval_batch(test_stack, w_arr.reshape(1, -1), hungarian=False)[0]
            print(f"  Greedy Top-1 (for ref): {test_metrics_greedy[0]*100:.1f}%", file=sys.stderr)
            print(f"  Greedy Top-5 (for ref): {test_metrics_greedy[1]*100:.1f}%", file=sys.stderr)

        w_str = {k: round(v, 4) for k, v in best_overall["weights"].items()}
        print(f"\n  Weights: {w_str}", file=sys.stderr)

        output = {
            "mode": "train_optimize_test_evaluate",
            "metric_scope": {
                "hungarian_top1": "Closed-set bipartite optimal assignment (Kuhn-Munkres). Requires N_query == N_candidate. NOT directly comparable to standard retrieval Top-1.",
                "greedy_top5": "Standard row-independent retrieval metric.",
            },
            "n_subsamples": args.n_subsamples,
            "best_train_weights": w_str,
            "test_metrics": {
                "top1_acc": float(test_metrics[0]),
                "top5_acc": float(test_metrics[1]),
            },
            "all_subsample_results": [
                {
                    "subsample_idx": r["subsample_idx"],
                    "train_top1": round(r["train_top1"], 4),
                    "train_top5": round(r["train_top5"], 4),
                    "weights": {k: round(v, 4) for k, v in r["weights"].items()},
                }
                for r in all_subsample_results
            ],
        }

        suffix = "hungarian" if use_hung else "greedy"
        out_path = results_dir / f"grid_search_honest_{suffix}_results.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to {out_path}", file=sys.stderr)

    else:
        print("=" * 80, file=sys.stderr)
        print("MODE: Test-optimize (original, for reference only)", file=sys.stderr)
        print("=" * 80, file=sys.stderr)

        print("Loading modality logits...", file=sys.stderr)
        modality_logits = load_modality_logits(results_dir, MODALITY_GLOBS_TEST, MODALITY_GLOBS_TEST_FALLBACK)
        print(f"Loaded {len(modality_logits)} modalities: {list(modality_logits.keys())}", file=sys.stderr)

        names = sorted(modality_logits.keys())
        modality_stack = torch.stack([modality_logits[nm] for nm in names], dim=0)

        all_candidates = run_optimization(modality_stack, names, hungarian=use_hung)

        print("\n" + "=" * 80, file=sys.stderr)
        best = all_candidates[0]
        w_str = {k: round(v, 4) for k, v in best["weights"].items()}
        print(f"BEST: top1={best['top1_acc']:.4f} top5={best['top5_acc']:.4f}", file=sys.stderr)
        print(f"  weights={w_str}", file=sys.stderr)

        cur_w = np.array([CURRENT_WEIGHTS.get(nm, 0.1) for nm in names], dtype=np.float64)
        cur_w /= cur_w.sum()
        cur_metrics = eval_batch(modality_stack, cur_w.reshape(1, -1), hungarian=use_hung)[0]

        output = {
            "mode": "test_optimize_original",
            "metric_scope": {
                "note": "Weights optimized directly on test set. For reference/upper-bound only, NOT a valid generalization estimate."
            },
            "current_weights": CURRENT_WEIGHTS,
            "current_metrics": {"top1_acc": float(cur_metrics[0]), "top5_acc": float(cur_metrics[1])},
            "best": {"top1_acc": best["top1_acc"], "top5_acc": best["top5_acc"], "weights": w_str},
            "top_results": [
                {"top1_acc": r["top1_acc"], "top5_acc": r["top5_acc"],
                 "weights": {k: round(v, 4) for k, v in r["weights"].items()}}
                for r in all_candidates[:10]
            ],
        }
        out_path = results_dir / "grid_search_results.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
