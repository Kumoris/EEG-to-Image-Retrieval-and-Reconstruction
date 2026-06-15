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
    m = Path(f"results/atms_vitl_seed{seed}_test_tta0.json")
    l = Path(f"results/atms_vitl_seed{seed}_test_tta0.logits.pt")
    if not (m.exists() and l.exists()):
        continue
    items.append({"seed": seed, **json.loads(m.read_text())})
    x = torch.load(l, map_location="cpu", weights_only=False)["logits"].float()
    x = (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True).clamp_min(1e-6)
    logits.append(x)

summary = {"per_seed": items, "summary": summarize_metric_dicts(items)}
if logits:
    ens = torch.stack(logits).mean(dim=0)
    summary["seed_ensemble"] = compute_retrieval_metrics(ens)
    torch.save({"logits": ens}, "results/atms_vitl_ensemble_logits.pt")
write_json("results/atms_vitl_summary.json", summary)
print(json.dumps(summary, indent=2))
PY
