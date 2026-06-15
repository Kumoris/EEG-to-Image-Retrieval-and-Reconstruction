from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .atm_s import ATM_S
from .data import deterministic_group_split, load_eeg_dataset, subset_records
from .encoders import build_eeg_encoder
from .features import features_for_ids, load_feature_cache
from .reconstruct_vae import VAELatentProjector, load_vae_decoder
from .utils import choose_device, ensure_dir, load_yaml, set_seed, write_json


class VAEPairDataset(Dataset):
    def __init__(self, eeg: torch.Tensor, vae_targets: torch.Tensor) -> None:
        self.eeg = eeg
        self.targets = vae_targets

    def __len__(self) -> int:
        return len(self.eeg)

    def __getitem__(self, idx: int):
        return self.eeg[idx], self.targets[idx]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EEG-to-VAE-latent projector")
    p.add_argument("--config", default="configs/atms_vae.yaml")
    p.add_argument("--data-dir", default="image-eeg-data")
    p.add_argument("--feature-cache", default="cache/features_multi.pt")
    p.add_argument("--feature-key", default="vae_feature")
    p.add_argument("--atm-ckpt", required=True, help="Path to pre-trained ATM-S VAE expert checkpoint")
    p.add_argument("--vae-name", default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--latent-channels", type=int, default=4)
    p.add_argument("--latent-spatial", type=int, default=64)
    p.add_argument("--projector-hidden", type=int, default=2048)
    p.add_argument("--projector-blocks", type=int, default=2)
    p.add_argument("--projector-drop", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="runs/vae_projector_seed0")
    p.add_argument("--device", default="auto")
    p.add_argument("--val-fraction", type=float, default=0.1)
    return p.parse_args()


@torch.no_grad()
def eval_projector(atm, projector: VAELatentProjector, vae_decoder, records, train_vae_flat, device, latent_channels=4, latent_spatial=64):
    atm.eval()
    projector.eval()
    eeg_loader = DataLoader(records.eeg, batch_size=128, shuffle=False, num_workers=0)
    mse_total = 0.0
    count = 0
    for eeg in eeg_loader:
        eeg = eeg.to(device)
        eeg_emb = F.normalize(atm(eeg), dim=-1)
        pred_latent = projector(eeg_emb)
        mse_total += F.mse_loss(pred_latent, torch.zeros_like(pred_latent)).item() * eeg.shape[0]
        count += eeg.shape[0]
    return {"val_mse_placeholder": mse_total / max(count, 1)}


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config) if Path(args.config).exists() else {}
    set_seed(args.seed)
    device = choose_device(args.device)
    out_dir = ensure_dir(args.output_dir)

    cache = load_feature_cache(args.feature_cache)
    if args.feature_key not in cache:
        available = [k for k in cache.keys() if k.endswith("_feature")]
        raise ValueError(f"Feature key '{args.feature_key}' not found in cache. Available: {available}")

    train_all = load_eeg_dataset(args.data_dir, "train", avg_trials=True, image_root="auto")
    tr_idx, va_idx = deterministic_group_split(train_all, args.val_fraction, args.seed)
    train_records = subset_records(train_all, tr_idx)
    val_records = subset_records(train_all, va_idx)

    vae_latent_dim = args.latent_channels * args.latent_spatial * args.latent_spatial
    ckpt = torch.load(args.atm_ckpt, map_location="cpu", weights_only=False)
    eeg_embed_dim = int(ckpt["embed_dim"])
    model_type = ckpt.get("model_type", "atm_s")
    model_cfg = ckpt.get("model_config", {})
    atm = build_eeg_encoder(model_type, int(ckpt["channels"]), int(ckpt["time_steps"]), embed_dim=eeg_embed_dim, **model_cfg).to(device)
    atm.load_state_dict(ckpt["model"])
    atm.eval()
    for p in atm.parameters():
        p.requires_grad = False

    vae_targets_flat = features_for_ids(cache, train_records.image_ids, args.feature_key)
    val_targets_flat = features_for_ids(cache, val_records.image_ids, args.feature_key)

    projector = VAELatentProjector(
        in_dim=eeg_embed_dim,
        vae_latent_dim=vae_targets_flat.shape[1],
        hidden_dim=args.projector_hidden,
        n_blocks=args.projector_blocks,
        drop=args.projector_drop,
    ).to(device)

    opt = torch.optim.AdamW(projector.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    train_embeds = []
    with torch.no_grad():
        for i in range(0, len(train_records.eeg), 256):
            batch = train_records.eeg[i:i+256].to(device)
            train_embeds.append(F.normalize(atm(batch), dim=-1).cpu())
    train_embeds = torch.cat(train_embeds, dim=0)

    dataset = VAEPairDataset(train_embeds, vae_targets_flat)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)

    history = []
    best_loss = float("inf")
    for epoch in range(args.epochs):
        projector.train()
        total_loss = 0.0
        count = 0
        for eeg_emb, tgt in loader:
            eeg_emb = eeg_emb.to(device)
            tgt = tgt.to(device)
            pred = projector(eeg_emb)
            loss = F.mse_loss(pred, tgt) + 0.5 * (1.0 - F.cosine_similarity(pred, tgt, dim=-1).mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * eeg_emb.shape[0]
            count += eeg_emb.shape[0]
        sched.step()
        avg_loss = total_loss / max(count, 1)
        row = {"epoch": epoch, "train_loss": avg_loss, "lr": float(opt.param_groups[0]["lr"])}
        history.append(row)
        print(f"epoch {epoch:03d}/{args.epochs-1:03d} loss={avg_loss:.4f}", flush=True)
        proj_ckpt = {
            "model": projector.state_dict(),
            "atm_ckpt": args.atm_ckpt,
            "feature_key": args.feature_key,
            "vae_latent_dim": vae_latent_dim,
            "hidden_dim": args.projector_hidden,
            "n_blocks": args.projector_blocks,
            "config": cfg,
            "metrics": row,
        }
        torch.save(proj_ckpt, out_dir / "last.pt")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(proj_ckpt, out_dir / "best.pt")
    write_json(out_dir / "metrics.json", history)
    print(f"Wrote projector checkpoints to {out_dir}", flush=True)


if __name__ == "__main__":
    main()