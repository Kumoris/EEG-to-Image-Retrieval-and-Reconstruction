from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import deterministic_group_split, load_eeg_dataset, subset_records
from .features import features_for_ids, load_feature_cache
from .fusion_atms import ATMFusionEncoder
from .train_atms import contrastive_loss, eval_records
from .utils import choose_device, compute_retrieval_metrics, ensure_dir, load_yaml, set_seed, write_json


class FusionPairDataset(Dataset):
    def __init__(self, eeg: torch.Tensor, targets: torch.Tensor) -> None:
        self.eeg = eeg
        self.targets = targets

    def __len__(self) -> int:
        return len(self.eeg)

    def __getitem__(self, idx: int):
        return self.eeg[idx], self.targets[idx]


@torch.no_grad()
def eval_fusion(model: ATMFusionEncoder, records, cache: dict, device: torch.device) -> dict[str, float]:
    model.eval()
    candidates = features_for_ids(cache, records.image_ids, "image_clean_feature").to(device)
    loader = DataLoader(records.eeg, batch_size=64, shuffle=False, num_workers=0)
    outs = []
    for eeg in loader:
        outs.append(F.normalize(model(eeg.to(device)), dim=-1).cpu())
    logits = torch.cat(outs, dim=0).to(device) @ candidates.T
    return compute_retrieval_metrics(logits.cpu())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/atms_fusion_vitl.yaml")
    p.add_argument("--data-dir", default="image-eeg-data")
    p.add_argument("--feature-cache", default="cache/features_vitl.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="runs/atms_fusion_vitl_seed0")
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--save-last-as-best", action="store_true")
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
    targets = features_for_ids(cache, train_records.image_ids, "image_clean_feature")
    embed_dim = int(targets.shape[1])
    model = ATMFusionEncoder(
        num_channels=int(train_records.eeg.shape[1]),
        time_dim=int(train_records.eeg.shape[2]),
        embed_dim=embed_dim,
        depth=int(cfg["fusion"]["depth"]),
        heads=int(cfg["fusion"]["heads"]),
        dropout=float(cfg["fusion"]["dropout"]),
        modality_dropout_p=float(cfg["fusion"]["modality_dropout_p"]),
        freeze_experts=bool(cfg["fusion"]["freeze_experts"]),
    ).to(device)
    model.load_expert_checkpoints(cfg["expert_ckpts"])
    logit_scale = torch.nn.Parameter(torch.tensor(math.log(1 / 0.07), dtype=torch.float32, device=device))
    params = [p for p in model.parameters() if p.requires_grad] + [logit_scale]
    opt = torch.optim.AdamW(params, lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(cfg["train"]["epochs"]), eta_min=float(cfg["train"].get("lr_min", 1e-6)))
    loader = DataLoader(
        FusionPairDataset(train_records.eeg, targets),
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
        for eeg, tgt in loader:
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
        sched.step()
        metrics = eval_fusion(model, val_records, cache, device)
        row = {"epoch": epoch, "train_loss": total / max(count, 1), "val_top1": metrics["top1_acc"], "val_top5": metrics["top5_acc"], "lr": float(opt.param_groups[0]["lr"])}
        history.append(row)
        print(f"epoch {epoch:03d}/{int(cfg['train']['epochs']) - 1:03d} loss={row['train_loss']:.4f} val@1={row['val_top1']:.4f} val@5={row['val_top5']:.4f}", flush=True)
        ckpt = {"model": model.state_dict(), "logit_scale": float(logit_scale.detach().cpu()), "channels": int(train_records.eeg.shape[1]), "time_steps": int(train_records.eeg.shape[2]), "embed_dim": embed_dim, "config": cfg, "metrics": row}
        torch.save(ckpt, out_dir / "last.pt")
        if row["val_top1"] > best_top1:
            best_top1 = row["val_top1"]
            torch.save(ckpt, out_dir / "best.pt")
    if args.save_last_as_best:
        torch.save(torch.load(out_dir / "last.pt", map_location="cpu", weights_only=False), out_dir / "best.pt")
    write_json(out_dir / "metrics.json", history)


if __name__ == "__main__":
    main()
