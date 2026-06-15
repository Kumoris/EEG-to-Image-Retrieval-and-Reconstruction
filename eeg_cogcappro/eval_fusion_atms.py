from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import load_eeg_dataset
from .features import features_for_ids, load_feature_cache
from .fusion_atms import ATMFusionEncoder
from .utils import choose_device, compute_retrieval_metrics, ensure_dir, safe_torch_load, write_csv, write_json


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="image-eeg-data")
    p.add_argument("--feature-cache", default="cache/features_vitl.pt")
    p.add_argument("--ckpt", default="runs/atms_fusion_vitl_seed0/best.pt")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--output", default="results/atms_fusion_vitl_seed0_test.json")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device = choose_device(args.device)
    ckpt = safe_torch_load(args.ckpt, map_location="cpu")
    cfg = ckpt["config"]
    records = load_eeg_dataset(args.data_dir, args.split, avg_trials=True, image_root="auto")
    cache = load_feature_cache(args.feature_cache)
    model = ATMFusionEncoder(
        int(ckpt["channels"]),
        int(ckpt["time_steps"]),
        int(ckpt["embed_dim"]),
        depth=int(cfg["fusion"]["depth"]),
        heads=int(cfg["fusion"]["heads"]),
        dropout=float(cfg["fusion"]["dropout"]),
        modality_dropout_p=0.0,
        freeze_experts=True,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    candidates = features_for_ids(cache, records.image_ids, "image_clean_feature").to(device)
    loader = DataLoader(records.eeg, batch_size=64, shuffle=False, num_workers=0)
    outs = []
    for eeg in loader:
        outs.append(F.normalize(model(eeg.to(device)), dim=-1).cpu())
    logits = torch.cat(outs, dim=0).to(device) @ candidates.T
    metrics = compute_retrieval_metrics(logits.cpu())
    output = Path(args.output)
    ensure_dir(output.parent)
    write_json(output, {"split": args.split, "ckpt": args.ckpt, **metrics})
    torch.save({"logits": logits.cpu(), "image_ids": records.image_ids, "concepts": records.concepts}, output.with_suffix(".logits.pt"))
    ranks = logits.cpu().topk(k=min(5, logits.shape[1]), dim=1)
    rows = []
    for i, image_id in enumerate(records.image_ids):
        row = {"index": i, "image_id": image_id, "gt_concept": records.concepts[i]}
        for j in range(ranks.indices.shape[1]):
            cand = int(ranks.indices[i, j])
            row[f"pred{j+1}"] = records.image_ids[cand]
            row[f"score{j+1}"] = float(ranks.values[i, j])
        rows.append(row)
    fields = ["index", "image_id", "gt_concept"] + [x for j in range(1, 6) for x in (f"pred{j}", f"score{j}")]
    write_csv(output.with_suffix(".top5.csv"), rows, fields)
    print(f"Wrote metrics: {output}", flush=True)


if __name__ == "__main__":
    main()
