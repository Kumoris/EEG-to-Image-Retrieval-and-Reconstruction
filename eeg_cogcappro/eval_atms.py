from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .atm_s import ATM_S
from .data import load_eeg_dataset
from .encoders import build_eeg_encoder
from .features import features_for_ids, load_feature_cache
from .transforms_eeg import EEGTrainTransform
from .utils import choose_device, compute_retrieval_metrics, ensure_dir, safe_torch_load, write_csv, write_json


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="image-eeg-data")
    p.add_argument("--feature-cache", default="cache/features_vitl.pt")
    p.add_argument("--feature-key", default=None, help="Feature cache key for evaluation. Defaults to 'image_clean_feature' for backward compat.")
    p.add_argument("--ckpt", default="runs/atms_vitl_seed0/best.pt")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--output", default="results/atms_vitl_seed0_test.json")
    p.add_argument("--tta-n", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device = choose_device(args.device)
    ckpt = safe_torch_load(args.ckpt, map_location="cpu")
    cache = load_feature_cache(args.feature_cache)
    feature_key = args.feature_key or ckpt.get("feature_key", "image_clean_feature")
    if feature_key not in cache:
        available = [k for k in cache.keys() if k.endswith("_feature") or k.startswith("image_")]
        raise ValueError(f"Feature key '{feature_key}' not in cache. Available: {available}")
    print(f"Evaluating with feature key: {feature_key}", flush=True)
    records = load_eeg_dataset(args.data_dir, args.split, avg_trials=True, image_root="auto")
    model_type = ckpt.get("model_type", "atm_s")
    model_cfg = ckpt.get("model_config", {})
    model = build_eeg_encoder(model_type, int(ckpt["channels"]), int(ckpt["time_steps"]), int(ckpt["embed_dim"]), **model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    candidates = features_for_ids(cache, records.image_ids, feature_key).to(device)
    loader = DataLoader(records.eeg, batch_size=128, shuffle=False, num_workers=0)
    aug = EEGTrainTransform(noise_std=0.01, channel_dropout_p=0.1, temporal_jitter=0, time_mask_frac=0.1)
    outs = []
    for eeg in loader:
        eeg = eeg.to(device)
        if args.tta_n > 0:
            preds = [F.normalize(model(aug(eeg.clone())), dim=-1) for _ in range(args.tta_n)]
            pred = F.normalize(torch.stack(preds).mean(0), dim=-1)
        else:
            pred = F.normalize(model(eeg), dim=-1)
        outs.append(pred.cpu())
    eeg_feats = torch.cat(outs, dim=0).to(device)
    logits = eeg_feats @ candidates.T
    metrics = compute_retrieval_metrics(logits.cpu())
    output = Path(args.output)
    ensure_dir(output.parent)
    write_json(output, {"split": args.split, "ckpt": args.ckpt, "feature_key": feature_key, **metrics})
    torch.save({"logits": logits.cpu(), "image_ids": records.image_ids, "concepts": records.concepts, "feature_key": feature_key}, output.with_suffix(".logits.pt"))
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
    print(f"Wrote metrics: {output} (feature_key={feature_key})", flush=True)


if __name__ == "__main__":
    main()
