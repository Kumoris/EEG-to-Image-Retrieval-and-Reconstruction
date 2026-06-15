#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import json
from pathlib import Path
import torch

from eeg_cogcappro.utils import compute_retrieval_metrics, summarize_metric_dicts, write_json

items = []
logits = []
for seed in range(10):
    path = Path(f"results/seed{seed}_test.json")
    if path.exists():
        items.append(json.loads(path.read_text()))
    lp = Path(f"results/seed{seed}_test.logits.pt")
    if lp.exists():
        obj = torch.load(lp, map_location="cpu", weights_only=False)
        x = obj["logits"].float()
        x = (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True).clamp_min(1e-6)
        logits.append(x)

summary = {"per_seed": items, "summary": summarize_metric_dicts(items) if items else {}}
if logits:
    ens = torch.stack(logits).mean(dim=0)
    summary["seed_ensemble"] = compute_retrieval_metrics(ens)
    torch.save({"logits": ens}, "results/ensemble_test_logits.pt")
Path("results").mkdir(exist_ok=True)
write_json("results/summary_10seeds.json", summary)
print("| metric | mean | std |")
print("|---|---:|---:|")
for k, v in summary.get("summary", {}).items():
    print(f"| {k} | {v['mean']:.4f} | {v['std']:.4f} |")
if "seed_ensemble" in summary:
    print(f"| ensemble_top1_acc | {summary['seed_ensemble']['top1_acc']:.4f} | 0.0000 |")
    print(f"| ensemble_top5_acc | {summary['seed_ensemble']['top5_acc']:.4f} | 0.0000 |")
PY
