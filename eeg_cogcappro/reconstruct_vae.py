from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .atm_s import ATM_S
from .data import load_eeg_dataset
from .encoders import build_eeg_encoder
from .features import features_for_ids, load_feature_cache
from .utils import choose_device, ensure_dir, l2_normalize, safe_torch_load, write_csv, write_json


class VAELatentProjector(nn.Module):
    def __init__(self, in_dim: int = 512, vae_latent_dim: int = 16384, hidden_dim: int = 2048, n_blocks: int = 2, drop: float = 0.3) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.vae_latent_dim = vae_latent_dim
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden_dim * 2, hidden_dim))
            for _ in range(n_blocks)
        ])
        self.proj_out = nn.Linear(hidden_dim, vae_latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        for block in self.blocks:
            x = x + block(x)
        return self.proj_out(x)


def load_vae_decoder(vae_name: str = "stabilityai/sd-vae-ft-mse", device: torch.device = torch.device("cpu")):
    try:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(vae_name, torch_dtype=torch.float32)
        vae = vae.eval().to(device)
        return vae
    except Exception as exc:
        print(f"Warning: Could not load VAE decoder ({vae_name}): {exc}", flush=True)
        return None


@torch.no_grad()
def decode_vae_latent(vae, latent_flat: torch.Tensor, projector: VAELatentProjector, device: torch.device, latent_channels: int = 4, spatial_size: int = 64, image_size: int = 256) -> list:
    projected = projector(latent_flat.to(device))
    latent_4d = projected.view(-1, latent_channels, spatial_size, spatial_size)
    decoded = vae.decode(latent_4d).sample
    decoded = F.interpolate(decoded, size=(image_size, image_size), mode="bilinear", align_corners=False)
    images = []
    for i in range(decoded.shape[0]):
        img = decoded[i].clamp(0, 1).cpu().permute(1, 2, 0).numpy()
        import numpy as np
        from PIL import Image
        img = Image.fromarray((img * 255).astype("uint8"))
        images.append(img)
    return images


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VAE-based reconstruction from ATM-S + VAE expert predictions")
    p.add_argument("--data-dir", default="image-eeg-data")
    p.add_argument("--feature-cache", default="cache/features_multi.pt")
    p.add_argument("--feature-key", default="vae_feature")
    p.add_argument("--vae-ckpt", default="runs/atms_vae_seed0/best.pt")
    p.add_argument("--retrieval-logits", default=None, help="Optional ensemble retrieval logits .pt for concept-constrained reconstruction")
    p.add_argument("--projector-ckpt", default=None, help="Optional pre-trained VAELatentProjector checkpoint")
    p.add_argument("--vae-name", default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--output-dir", default="recons/vae_seed0")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--latent-channels", type=int, default=4)
    p.add_argument("--latent-spatial", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--method", default="vae_decode", choices=["vae_decode", "vae_train_nearest", "hybrid"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    out_dir = ensure_dir(args.output_dir)

    cache = load_feature_cache(args.feature_cache)
    if args.feature_key not in cache:
        available = [k for k in cache.keys() if k.endswith("_feature")]
        raise ValueError(f"Feature key '{args.feature_key}' not found. Available: {available}")

    vae_decoder = load_vae_decoder(args.vae_name, device)
    if vae_decoder is None and args.method != "vae_train_nearest":
        print("Falling back to vae_train_nearest method since VAE decoder is not available", flush=True)
        args.method = "vae_train_nearest"

    test = load_eeg_dataset(args.data_dir, "test", avg_trials=True, image_root="auto")
    train = load_eeg_dataset(args.data_dir, "train", avg_trials=True, image_root="auto")

    ckpt = safe_torch_load(args.vae_ckpt, map_location="cpu")
    eeg_embed_dim = int(ckpt["embed_dim"])
    model_type = ckpt.get("model_type", "atm_s")
    model_cfg = ckpt.get("model_config", {})
    atm = build_eeg_encoder(model_type, int(ckpt["channels"]), int(ckpt["time_steps"]), embed_dim=eeg_embed_dim, **model_cfg).to(device)
    atm.load_state_dict(ckpt["model"])
    atm.eval()

    vae_latent_dim = args.latent_channels * args.latent_spatial * args.latent_spatial
    projector = None
    can_decode = False
    if args.projector_ckpt and Path(args.projector_ckpt).exists():
        proj_ckpt = safe_torch_load(args.projector_ckpt, map_location="cpu")
        hidden_dim = int(proj_ckpt.get("hidden_dim", 2048))
        n_blocks = int(proj_ckpt.get("n_blocks", 2))
        actual_out_dim = int(proj_ckpt["model"]["proj_out.bias"].shape[0])
        projector = VAELatentProjector(in_dim=eeg_embed_dim, vae_latent_dim=actual_out_dim,
                                        hidden_dim=hidden_dim, n_blocks=n_blocks).to(device)
        projector.load_state_dict(proj_ckpt["model"])
        projector.eval()
        can_decode = (actual_out_dim == vae_latent_dim)
    else:
        print("Warning: No pre-trained VAELatentProjector provided.", flush=True)

    train_vae_feats = features_for_ids(cache, train.image_ids, args.feature_key)

    eeg_loader = DataLoader(test.eeg, batch_size=64, shuffle=False)
    eeg_embeds = []
    for batch in eeg_loader:
        eeg_embeds.append(F.normalize(atm(batch.to(device)), dim=-1).cpu())
    eeg_embeds = torch.cat(eeg_embeds, dim=0)

    from .image_resolver import load_rgb
    from .reconstruct import LEAKAGE_POLICY

    if args.method == "vae_decode" and vae_decoder is not None and can_decode and projector is not None:
        latent_flat = projector(eeg_embeds.to(device))
        images = decode_vae_latent(vae_decoder, latent_flat, projector, device,
                                    latent_channels=args.latent_channels,
                                    spatial_size=args.latent_spatial,
                                    image_size=args.image_size)
        rows = []
        for i, img in enumerate(images):
            dst = out_dir / f"{i:03d}.png"
            img.save(dst)
            rows.append({"query_index": i, "query_image_id": test.image_ids[i],
                          "source_kind": "vae_decode", "leakage_policy": LEAKAGE_POLICY})
        _write_rows(out_dir, rows, {"method": "vae_decode", "num_images": len(rows), "leakage_policy": LEAKAGE_POLICY})
        return

    if not can_decode:
        print("Falling back to vae_train_nearest (projector output dim != VAE latent dim).", flush=True)

    sims = eeg_embeds @ train_vae_feats.T.to(eeg_embeds.device)
    nearest = sims.argmax(dim=1)
    rows = []
    for i in range(len(test.image_ids)):
        train_idx = int(nearest[i].item())
        dst = out_dir / f"{i:03d}.png"
        src = train.image_paths[train_idx]
        if src is not None and src.exists():
            img = load_rgb(src).resize((args.image_size, args.image_size))
            img.save(dst)
            source_kind = "vae_train_nearest"
        else:
            import hashlib as _hl
            from PIL import Image as _Img, ImageDraw as _Dr
            digest = _hl.sha256(train.image_ids[train_idx].encode()).digest()
            img = _Img.new("RGB", (args.image_size, args.image_size), tuple(int(v) for v in digest[:3]))
            _Dr.Draw(img).text((10, 10), train.image_ids[train_idx][:28], fill=(255, 255, 255))
            img.save(dst)
            source_kind = "prompt_placeholder"
        rows.append({"query_index": i, "query_image_id": test.image_ids[i],
                      "nearest_train_index": train_idx, "nearest_train_image_id": train.image_ids[train_idx],
                      "nearest_train_concept": train.concepts[train_idx],
                      "source_kind": source_kind, "leakage_policy": LEAKAGE_POLICY})
    _write_rows(out_dir, rows, {"method": "vae_train_nearest", "num_images": len(rows), "leakage_policy": LEAKAGE_POLICY})


def _write_rows(out_dir, rows, summary):
    fields = list(rows[0].keys()) if rows else []
    write_csv(out_dir / "manifest.csv", rows, fields)
    write_json(out_dir / "summary.json", summary)
    print(f"Wrote {len(rows)} VAE reconstructions to {out_dir}", flush=True)


if __name__ == "__main__":
    main()