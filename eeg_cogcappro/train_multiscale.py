from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import deterministic_group_split, load_eeg_dataset, subset_records
from .features import load_feature_cache
from .multiscale_blur import MultiScaleBlurDataset, MultiscaleBlurModel
from .transforms_eeg import EEGTrainTransform
from .utils import choose_device, compute_retrieval_metrics, ensure_dir, load_yaml, set_seed, write_json


def contrastive_loss(eeg_emb: torch.Tensor, vis_emb: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    eeg_emb = F.normalize(eeg_emb, dim=-1)
    vis_emb = F.normalize(vis_emb, dim=-1)
    logits = logit_scale * eeg_emb @ vis_emb.T
    targets = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))


@torch.no_grad()
def eval_multiscale(
    model: torch.nn.Module,
    records,
    cache: dict,
    feature_keys: list[str],
    device: torch.device,
    tta_n: int = 0,
) -> dict[str, float]:
    model.eval()
    dataset = MultiScaleBlurDataset(records, cache, feature_keys, augment=False)
    all_eeg = []
    all_vis = []
    aug = EEGTrainTransform(noise_std=0.01, channel_dropout_p=0.1, temporal_jitter=0, time_mask_frac=0.1)
    for eeg, scale_feats, _label in DataLoader(dataset, batch_size=128, shuffle=False):
        eeg = eeg.to(device)
        scale_feats = scale_feats.to(device)
        if tta_n > 0:
            preds_eeg = []
            for _ in range(tta_n):
                eeg_aug = aug(eeg.clone())
                eeg_emb, vis_emb = model(eeg_aug, scale_feats)
                preds_eeg.append(F.normalize(eeg_emb, dim=-1))
            eeg_emb = F.normalize(torch.stack(preds_eeg).mean(0), dim=-1)
            vis_emb = F.normalize(vis_emb, dim=-1)
        else:
            eeg_emb, vis_emb = model(eeg, scale_feats)
            eeg_emb = F.normalize(eeg_emb, dim=-1)
            vis_emb = F.normalize(vis_emb, dim=-1)
        all_eeg.append(eeg_emb.cpu())
        all_vis.append(vis_emb.cpu())
    logits = torch.cat(all_eeg, dim=0).to(device) @ torch.cat(all_vis, dim=0).T.to(device)
    return compute_retrieval_metrics(logits.cpu())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/atms_multiscale_blur.yaml")
    p.add_argument("--data-dir", default="image-eeg-data")
    p.add_argument("--feature-cache", default=None)
    p.add_argument("--feature-keys", nargs="+", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="runs/multiscale_blur_seed0")
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--save-last-as-best", action="store_true")
    p.add_argument("--visual-lr-scale", type=float, default=None, help="LR scale for visual encoder relative to EEG encoder. Defaults to config value or 0.1.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    feat_cfg = cfg.get("features", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})

    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs

    set_seed(args.seed)
    device = choose_device(args.device)
    out_dir = ensure_dir(args.output_dir)

    feature_cache_path = args.feature_cache or feat_cfg.get("feature_cache", "cache/features_vitl_real.pt")
    feature_keys = args.feature_keys or feat_cfg.get("feature_keys", ["image_clean_feature", "image_fovea_low", "image_fovea_mid", "image_fovea_high"])
    feature_dim = int(feat_cfg.get("feature_dim", 768))

    cache = load_feature_cache(feature_cache_path)
    for k in feature_keys:
        if k not in cache:
            raise ValueError(f"Feature key '{k}' not in cache. Available: {[x for x in cache if 'feature' in x or 'fovea' in x]}")
    print(f"Using feature keys: {feature_keys} (dim={feature_dim}, n_scales={len(feature_keys)})", flush=True)

    train_all = load_eeg_dataset(args.data_dir, "train", avg_trials=True, image_root="auto")
    tr_idx, va_idx = deterministic_group_split(train_all, float(train_cfg.get("val_fraction", 0.1)), args.seed)
    train_records = subset_records(train_all, tr_idx)
    val_records = subset_records(train_all, va_idx)

    n_scales = len(feature_keys)
    n_channels = train_records.eeg.shape[1]
    time_steps = train_records.eeg.shape[2]
    embed_dim = feature_dim

    model = MultiscaleBlurModel(
        num_channels=n_channels,
        time_dim=time_steps,
        n_scales=n_scales,
        feature_dim=feature_dim,
        embed_dim=embed_dim,
        eeg_attn_heads=int(model_cfg.get("eeg_attn_heads", 8)),
        eeg_attn_depth=int(model_cfg.get("eeg_attn_depth", 2)),
        visual_encoder_type=model_cfg.get("visual_encoder_type", "linear"),
        visual_dropout=float(model_cfg.get("visual_dropout", 0.1)),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    eeg_params = sum(p.numel() for p in model.eeg_encoder.parameters())
    vis_params = sum(p.numel() for p in model.visual_encoder.parameters())
    print(f"Model parameters: {n_params:,} (EEG: {eeg_params:,}, Visual: {vis_params:,})", flush=True)

    train_dataset = MultiScaleBlurDataset(
        train_records, cache, feature_keys, augment=bool(train_cfg.get("eeg_aug", True)),
    )
    val_dataset = MultiScaleBlurDataset(
        val_records, cache, feature_keys, augment=False,
    )

    logit_scale = torch.nn.Parameter(torch.tensor(math.log(1 / 0.07), dtype=torch.float32, device=device))
    visual_lr_scale = float(args.visual_lr_scale) if args.visual_lr_scale is not None else float(train_cfg.get("visual_lr_scale", 0.1))
    param_groups = [
        {"params": list(model.eeg_encoder.parameters()) + [logit_scale], "lr": float(train_cfg["lr"])},
        {"params": list(model.visual_encoder.parameters()), "lr": float(train_cfg["lr"]) * visual_lr_scale},
    ]
    opt = torch.optim.AdamW(param_groups, weight_decay=float(train_cfg["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(train_cfg["epochs"]), eta_min=float(train_cfg.get("lr_min", 1e-6)))
    lam = float(train_cfg.get("lambda_clip", 0.9))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=True,
        drop_last=True,
    )

    history = []
    best_top1 = -1.0

    for epoch in range(int(train_cfg["epochs"])):
        model.train()
        total_loss = 0.0
        count = 0
        last_loss = 0.0
        for eeg, scale_feats, _label in train_loader:
            eeg = eeg.to(device, non_blocking=True)
            scale_feats = scale_feats.to(device, non_blocking=True)
            eeg_emb, vis_emb = model(eeg, scale_feats)
            loss_con = contrastive_loss(eeg_emb, vis_emb, logit_scale.exp().clamp(max=100.0))
            loss_mse = F.mse_loss(F.normalize(eeg_emb, dim=-1), F.normalize(vis_emb, dim=-1))
            loss = lam * loss_con + (1.0 - lam) * loss_mse
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item()) * eeg.shape[0]
            count += eeg.shape[0]
            last_loss = float(loss.item())

        sched.step()
        metrics = eval_multiscale(model, val_records, cache, feature_keys, device, tta_n=0)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(count, 1),
            "last_loss": last_loss,
            "val_top1": metrics["top1_acc"],
            "val_top5": metrics["top5_acc"],
            "lr_eeg": float(opt.param_groups[0]["lr"]),
            "lr_vis": float(opt.param_groups[1]["lr"]),
            "logit_scale": float(logit_scale.exp().detach().cpu()),
            "feature_keys": feature_keys,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{int(train_cfg['epochs']) - 1:03d} "
            f"loss={row['train_loss']:.4f} val@1={row['val_top1']:.4f} val@5={row['val_top5']:.4f} "
            f"lr_eeg={row['lr_eeg']:.6f} lr_vis={row['lr_vis']:.6f}",
            flush=True,
        )

        ckpt = {
            "model": model.state_dict(),
            "logit_scale": float(logit_scale.detach().cpu()),
            "channels": n_channels,
            "time_steps": time_steps,
            "embed_dim": embed_dim,
            "n_scales": n_scales,
            "feature_dim": feature_dim,
            "feature_keys": feature_keys,
            "model_type": model_cfg.get("visual_encoder_type", "linear"),
            "model_config": {k: v for k, v in model_cfg.items() if k != "type"},
            "config": cfg,
            "visual_lr_scale": visual_lr_scale,
            "seed": args.seed,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if row["val_top1"] > best_top1:
            best_top1 = row["val_top1"]
            torch.save(ckpt, out_dir / "best.pt")

    if args.save_last_as_best:
        last = torch.load(out_dir / "last.pt", map_location="cpu", weights_only=False)
        torch.save(last, out_dir / "best.pt")

    write_json(out_dir / "metrics.json", history)
    print(f"Training complete. Best val@1={best_top1:.4f}", flush=True)


if __name__ == "__main__":
    main()