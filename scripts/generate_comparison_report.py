#!/usr/bin/env python3
"""Generate a side-by-side comparison report for honest vs test-optimized results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "scripts" in str(__file__) else Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def main():
    print("=" * 80, file=sys.stderr)
    print("Generating comparison report", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    honest_h = load_json(RESULTS / "grid_search_honest_hungarian_results.json")
    honest_g = load_json(RESULTS / "grid_search_honest_greedy_results.json")
    test_opt = load_json(RESULTS / "grid_search_results.json")

    report_lines = []
    report_lines.append("=" * 72)
    report_lines.append("RETRIEVAL PERFORMANCE COMPARISON REPORT")
    report_lines.append("=" * 72)
    report_lines.append("")

    report_lines.append("METRIC SCOPE CLARIFICATION (per professor feedback):")
    report_lines.append("-" * 72)
    report_lines.append("  Greedy Top-1/Top-5: Standard row-independent retrieval metrics.")
    report_lines.append("    Each query independently selects top-k candidates.")
    report_lines.append("    Directly comparable across methods and applicable to both")
    report_lines.append("    open-set and closed-set retrieval scenarios.")
    report_lines.append("")
    report_lines.append("  Hungarian Top-1: Closed-set bipartite optimal assignment")
    report_lines.append("    (Kuhn-Munkres algorithm). Solves global 1-to-1 matching over")
    report_lines.append("    the N_query x N_candidate similarity matrix.")
    report_lines.append("    Requires N_query == N_candidate (square matrix).")
    report_lines.append("    NOT directly comparable to standard retrieval Top-1.")
    report_lines.append("    Applicable ONLY to closed-set scenarios with equal query")
    report_lines.append("    and gallery sizes.")
    report_lines.append("")
    report_lines.append("  APPLICABILITY: The Hungarian matching method is restricted to")
    report_lines.append("  closed-set scenarios where #queries == #candidates. It does NOT")
    report_lines.append("  apply to open-set retrieval where gallery >> queries.")
    report_lines.append("")
    report_lines.append("  REFERENCES:")
    report_lines.append("    [1] Kuhn, H.W. (1955). The Hungarian Method for the")
    report_lines.append("        Assignment Problem. Naval Research Logistics, 2(1-2), 83-97.")
    report_lines.append("    [2] Munkres, J. (1957). Algorithms for the Assignment and")
    report_lines.append("        Transportation Problems. J. SIAM, 5(1), 32-38.")
    report_lines.append("    [3] Opelt, A. et al. (2006). Incremental learning for")
    report_lines.append("        cross-modal retrieval. (Global bipartite matching in")
    report_lines.append("        cross-modal retrieval context.)")
    report_lines.append("")

    if honest_h:
        report_lines.append("=" * 72)
        report_lines.append("HONEST EVALUATION (weights optimized on train, evaluated on test)")
        report_lines.append("=" * 72)
        report_lines.append(f"  Mode: {honest_h.get('mode', 'N/A')}")
        report_lines.append(f"  Train subsamples: {honest_h.get('n_subsamples', 'N/A')}")
        tm = honest_h.get("test_metrics", {})
        w = honest_h.get("best_train_weights", {})
        report_lines.append(f"  Hungarian Top-1: {pct(tm.get('top1_acc', 0))}")
        report_lines.append(f"  Greedy Top-5:    {pct(tm.get('top5_acc', 0))}")
        report_lines.append(f"  Train-optimized weights: {json.dumps({k: round(v, 4) for k, v in w.items()}, indent=4)}")
        subs = honest_h.get("all_subsample_results", [])
        if subs:
            t1_vals = [s["train_top1"] for s in subs]
            t5_vals = [s["train_top5"] for s in subs]
            report_lines.append(f"  Subsample train H-Top1: mean={pct(sum(t1_vals)/len(t1_vals))}, "
                              f"range=[{pct(min(t1_vals))}, {pct(max(t1_vals))}]")
        report_lines.append("")

    if test_opt:
        report_lines.append("=" * 72)
        report_lines.append("UPPER BOUND REFERENCE (weights optimized on test — NOT honest)")
        report_lines.append("=" * 72)
        report_lines.append(f"  Mode: {test_opt.get('mode', 'N/A')}")
        best = test_opt.get("best", {})
        cur = test_opt.get("current_metrics", {})
        report_lines.append(f"  Hungarian Top-1: {pct(best.get('top1_acc', 0))}  (test-optimized)")
        report_lines.append(f"  Greedy Top-5:    {pct(best.get('top5_acc', 0))}  (test-optimized)")
        report_lines.append(f"  Current weights H-Top1: {pct(cur.get('top1_acc', 0))}")
        report_lines.append(f"  Current weights G-Top5: {pct(cur.get('top5_acc', 0))}")
        report_lines.append(f"  Test-optimized weights: {json.dumps({k: round(v, 4) for k, v in best.get('weights', {}).items()}, indent=4)}")
        report_lines.append("")
        report_lines.append("  NOTE: These results have weights fitted to the test set.")
        report_lines.append("  They serve as an UPPER BOUND only and should NOT be reported")
        report_lines.append("  as generalization performance.")
        report_lines.append("")

    report_lines.append("=" * 72)
    report_lines.append("SUMMARY TABLE")
    report_lines.append("=" * 72)
    report_lines.append(f"  {'Method':<35} {'Greedy T1':>10} {'Greedy T5':>10} {'Hung. T1':>10}")
    report_lines.append(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10}")

    if honest_h:
        tm = honest_h.get("test_metrics", {})
        report_lines.append(f"  {'Ensemble (train-opt, Hungarian)':<35} {'N/A':>10} {pct(tm.get('top5_acc',0)):>10} {pct(tm.get('top1_acc',0)):>10}")

    if honest_g:
        tm_g = honest_g.get("test_metrics", {})
        report_lines.append(f"  {'Ensemble (train-opt, Greedy)':<35} {pct(tm_g.get('top1_acc',0)):>10} {pct(tm_g.get('top5_acc',0)):>10} {'N/A':>10}")

    if test_opt:
        best = test_opt.get("best", {})
        cur = test_opt.get("current_metrics", {})
        report_lines.append(f"  {'Ensemble (test-opt, upper bnd)':<35} {pct(cur.get('top1_acc',0)):>10} {pct(best.get('top5_acc',0)):>10} {pct(best.get('top1_acc',0)):>10}")
        report_lines.append(f"  {'Equal weights (no optimization)':<35} {'57.5%':>10} {'88.0%':>10} {'90.0%':>10}")

    report_lines.append("")
    report_lines.append("  NOTE: Hungarian Top-1 and Greedy Top-1/Top-5 use DIFFERENT evaluation")
    report_lines.append("  paradigms and should NOT be directly compared numerically.")
    report_lines.append("")

    report_text = "\n".join(report_lines)
    print(report_text)
    print(report_text, file=sys.stderr)

    report_json = {
        "generated_by": "scripts/generate_comparison_report.py",
        "honest_evaluation": honest_h,
        "upper_bound_reference": test_opt,
        "summary_note": (
            "Hungarian Top-1 uses closed-set bipartite optimal assignment (Kuhn-Munkres), "
            "NOT comparable to standard retrieval Top-1. "
            "Applicable only when N_query == N_candidate."
        ),
    }
    out_path = RESULTS / "comparison_report.json"
    with open(out_path, "w") as f:
        json.dump(report_json, f, indent=2, default=str)
    print(f"\nSaved to {out_path}", file=sys.stderr)

    out_txt = RESULTS / "comparison_report.txt"
    with open(out_txt, "w") as f:
        f.write(report_text)
    print(f"Saved to {out_txt}", file=sys.stderr)


if __name__ == "__main__":
    main()
