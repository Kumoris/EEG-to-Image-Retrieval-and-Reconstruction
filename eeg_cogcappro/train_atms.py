from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .atm_s import ATM_S
from .data import EEGRecords, deterministic_group_split, load_eeg_dataset, subset_records
from .encoders import build_eeg_encoder
from .features import features_for_ids, load_feature_cache
from .transforms_eeg import EEGTrainTransform
from .utils import choose_device, compute_retrieval_metrics, ensure_dir, load_yaml, set_seed, write_json


class PairDataset(Dataset):
    def __init__(self, records: EEGRecords, targets: torch.Tensor, augment: bool) -> None:
        self.records = records
        self.targets = targets
        self.transform = EEGTrainTransform(noise_std=0.01, channel_dropout_p=0.1, temporal_jitter=0, time_mask_frac=0.1) if augment else None

    def __len__(self) -> int:
        return len(self.records.eeg)

    def __getitem__(self, idx: int):
        eeg = self.records.eeg[idx]
        if self.transform is not None:
            eeg = self.transform(eeg)
        return eeg, self.targets[idx], self.records.labels[idx]


def contrastive_loss(eeg_emb: torch.Tensor, img_emb: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    eeg_emb = F.normalize(eeg_emb, dim=-1)
    img_emb = F.normalize(img_emb, dim=-1)
    logits = logit_scale * eeg_emb @ img_emb.T
    targets = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))


@torch.no_grad()
def eval_records(model: nn.Module, records: EEGRecords, cache: dict, device: torch.device, tta_n: int = 0, feature_key: str = "image_clean_feature") -> dict[str, float]:
    model.eval()
    candidates = features_for_ids(cache, records.image_ids, feature_key).to(device)
    loader = DataLoader(records.eeg, batch_size=128, shuffle=False, num_workers=0)
    aug = EEGTrainTransform(noise_std=0.01, channel_dropout_p=0.1, temporal_jitter=0, time_mask_frac=0.1)
    outs = []
    for eeg in loader:
        eeg = eeg.to(device)
        if tta_n > 0:
            preds = [F.normalize(model(aug(eeg.clone())), dim=-1) for _ in range(tta_n)]
            pred = F.normalize(torch.stack(preds).mean(0), dim=-1)
        else:
            pred = F.normalize(model(eeg), dim=-1)
        outs.append(pred.cpu())
    logits = torch.cat(outs, dim=0).to(device) @ candidates.T
    return compute_retrieval_metrics(logits.cpu())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/atms_vitl.yaml")
    p.add_argument("--data-dir", default="image-eeg-data")
    p.add_argument("--feature-cache", default="cache/features_vitl.pt")
    p.add_argument("--feature-key", default=None, help="Feature cache key for training targets. Defaults to 'image_clean_feature' for backward compat.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="runs/atms_vitl_seed0")
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--model", default=None, help="Encoder type: atm_s (default) or conformer")
    p.add_argument("--save-last-as-best", action="store_true", help="Use fixed final epoch as final checkpoint instead of train-derived val best.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    set_seed(args.seed)
    device = choose_device(args.device)
    out_dir = ensure_dir(args.output_dir)
    cache = load_feature_cache(args.feature_cache)
    train_all = load_eeg_dataset(args.data_dir, "train", avg_trials=True, image_root="auto")
    tr_idx, va_idx = deterministic_group_split(train_all, float(cfg["train"].get("val_fraction", 0.1)), args.seed)
    train_records = subset_records(train_all, tr_idx)
    val_records = subset_records(train_all, va_idx)
    feature_key = args.feature_key or cfg.get("features", {}).get("feature_cache_key", "image_clean_feature")
    if feature_key not in cache:
        available = [k for k in cache.keys() if k.endswith("_feature") or k.startswith("image_")]
        raise ValueError(f"Feature key '{feature_key}' not in cache. Available feature keys: {available}")
    print(f"Using feature key: {feature_key} (dim={cache[feature_key].shape[1]})", flush=True)
    train_targets = features_for_ids(cache, train_records.image_ids, feature_key)
    embed_dim = int(train_targets.shape[1])
    model_type = args.model or cfg.get("model", {}).get("type", "atm_s")
    model_cfg = {k: v for k, v in cfg.get("model", {}).items() if k != "type"}
    model = build_eeg_encoder(model_type, train_records.eeg.shape[1], train_records.eeg.shape[2], embed_dim=embed_dim, **model_cfg).to(device)
    logit_scale = torch.nn.Parameter(torch.tensor(math.log(1 / 0.07), dtype=torch.float32, device=device))
    params = list(model.parameters()) + [logit_scale]
    opt = torch.optim.AdamW(params, lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(cfg["train"]["epochs"]), eta_min=float(cfg["train"].get("lr_min", 1e-6)))
    loader = DataLoader(
        PairDataset(train_records, train_targets, augment=bool(cfg["train"].get("eeg_aug", True))),
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        pin_memory=True,
        drop_last=True,
    )
    history = []
    best_top1 = -1.0
    for epoch in range(int(cfg["train"]["epochs"])):
        model.train()
        total = 0.0
        count = 0
        last_loss = 0.0
        for eeg, tgt, _label in loader:
            eeg = eeg.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            pred = model(eeg)
            loss_con = contrastive_loss(pred, tgt, logit_scale.exp().clamp(max=100.0))
            loss_mse = F.mse_loss(F.normalize(pred, dim=-1), F.normalize(tgt, dim=-1))
            lam = float(cfg["train"].get("lambda_clip", 0.9))
            loss = lam * loss_con + (1.0 - lam) * loss_mse
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            total += float(loss.item()) * eeg.shape[0]
            count += eeg.shape[0]
            last_loss = float(loss.item())
        sched.step()
        metrics = eval_records(model, val_records, cache, device, tta_n=0, feature_key=feature_key)
        row = {
            "epoch": epoch,
            "train_loss": total / max(count, 1),
            "last_loss": last_loss,
            "val_top1": metrics["top1_acc"],
            "val_top5": metrics["top5_acc"],
            "lr": float(opt.param_groups[0]["lr"]),
            "logit_scale": float(logit_scale.exp().detach().cpu()),
            "feature_key": feature_key,
        }
        history.append(row)
        print(f"epoch {epoch:03d}/{int(cfg['train']['epochs']) - 1:03d} loss={row['train_loss']:.4f} val@1={row['val_top1']:.4f} val@5={row['val_top5']:.4f}", flush=True)
        ckpt = {
            "model": model.state_dict(),
            "logit_scale": float(logit_scale.detach().cpu()),
            "channels": int(train_records.eeg.shape[1]),
            "time_steps": int(train_records.eeg.shape[2]),
            "embed_dim": embed_dim,
            "model_type": model_type,
            "model_config": model_cfg,
            "config": cfg,
            "feature_key": feature_key,
            "metrics": row,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if row["val_top1"] > best_top1:
            best_top1 = row["val_top1"]
            torch.save(ckpt, out_dir / "best.pt")
    if args.save_last_as_best:
        last = torch.load(out_dir / "last.pt", map_location="cpu", weights_only=False)
        torch.save(last, out_dir / "best.pt")
    write_json(out_dir / "metrics.json", history)


if __name__ == "__main__":
    main()
