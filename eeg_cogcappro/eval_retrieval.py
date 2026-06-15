from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import EEGDataset, load_eeg_dataset, train_eeg_stats
from .features import features_for_ids, load_feature_cache
from .models import build_model_from_checkpoint
from .utils import choose_device, compute_retrieval_metrics, ensure_dir, safe_torch_load, write_csv, write_json


@torch.no_grad()
def compute_logits(model, records, cache: dict, device: torch.device, weights: dict[str, float]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    model.eval()
    candidates = features_for_ids(cache, records.image_ids, "image_clean_feature").to(device)
    loader = DataLoader(EEGDataset(records), batch_size=512, shuffle=False, num_workers=0)
    parts = {"img": [], "fusion": [], "aligned": [], "ensemble": []}
    for batch in loader:
        out = model(batch["eeg"].to(device))
        img = out["experts"]["img"] @ candidates.T
        fusion = out["fusion"] @ candidates.T
        aligned = out["aligned"]["fusion"] @ candidates.T
        ens = weights.get("img", 0.3) * img + weights.get("fusion", 0.5) * fusion + weights.get("aligned", 0.2) * aligned
        for k, v in [("img", img), ("fusion", fusion), ("aligned", aligned), ("ensemble", ens)]:
            parts[k].append(v.cpu())
    out_parts = {k: torch.cat(v, dim=0) for k, v in parts.items()}
    return out_parts["ensemble"], out_parts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="image-eeg-data")
    p.add_argument("--feature-cache", default="cache/features_rn50.pt")
    p.add_argument("--ckpt", default="runs/seed0/best.pt")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--output", default="results/seed0_test.json")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    ckpt = safe_torch_load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("config", {})
    cache = load_feature_cache(args.feature_cache)
    train_ref = load_eeg_dataset(args.data_dir, "train", avg_trials=bool(cfg.get("data", {}).get("avg_trials_train", False)), selected_channels=cfg.get("data", {}).get("selected_channels"), image_root="auto")
    mean, std = train_eeg_stats(train_ref)
    records = load_eeg_dataset(args.data_dir, args.split, avg_trials=True, selected_channels=cfg.get("data", {}).get("selected_channels"), image_root="auto").normalize(mean, std)
    model = build_model_from_checkpoint(ckpt, device)
    weights = cfg.get("eval", {}).get("ensemble_weights", {"img": 0.3, "fusion": 0.5, "aligned": 0.2})
    logits, parts = compute_logits(model, records, cache, device, weights)
    metrics = compute_retrieval_metrics(logits) if logits.shape[0] == logits.shape[1] else {}
    output = Path(args.output)
    ensure_dir(output.parent)
    write_json(output, {"split": args.split, "ckpt": args.ckpt, **metrics})
    torch.save({"logits": logits, "parts": parts, "image_ids": records.image_ids, "concepts": records.concepts}, output.with_suffix(".logits.pt"))
    ranks = logits.topk(k=min(5, logits.shape[1]), dim=1)
    rows = []
    for i, image_id in enumerate(records.image_ids):
        row = {"index": i, "image_id": image_id, "gt_concept": records.concepts[i]}
        for j in range(ranks.indices.shape[1]):
            cand = int(ranks.indices[i, j])
            row[f"pred{j+1}"] = records.image_ids[cand]
            row[f"score{j+1}"] = float(ranks.values[i, j].item())
        rows.append(row)
    fields = ["index", "image_id", "gt_concept"] + [x for j in range(1, 6) for x in (f"pred{j}", f"score{j}")]
    write_csv(output.with_suffix(".top5.csv"), rows, fields)
    print(f"Wrote metrics: {output}", flush=True)


if __name__ == "__main__":
    main()
