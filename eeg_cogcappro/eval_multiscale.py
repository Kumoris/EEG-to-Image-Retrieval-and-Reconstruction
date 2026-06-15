from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import load_eeg_dataset
from .features import load_feature_cache
from .multiscale_blur import MultiScaleBlurDataset, MultiscaleBlurModel
from .transforms_eeg import EEGTrainTransform
from .utils import choose_device, compute_retrieval_metrics, ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to best.pt or last.pt checkpoint")
    p.add_argument("--data-dir", default="image-eeg-data")
    p.add_argument("--feature-cache", default=None)
    p.add_argument("--feature-keys", nargs="+", default=None)
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--tta", type=int, default=0, help="Test-time augmentation rounds (0=no TTA)")
    p.add_argument("--output", default=None, help="Output .logits.pt file path")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=128)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    feature_keys = args.feature_keys or ckpt.get("feature_keys", ["image_clean_feature", "image_fovea_low", "image_fovea_mid", "image_fovea_high"])
    n_scales = ckpt.get("n_scales", len(feature_keys))
    feature_dim = ckpt.get("feature_dim", 768)
    embed_dim = ckpt.get("embed_dim", 768)
    n_channels = ckpt["channels"]
    time_steps = ckpt["time_steps"]
    model_type = ckpt.get("model_type", "linear")
    model_config = ckpt.get("model_config", {})

    model = MultiscaleBlurModel(
        num_channels=n_channels,
        time_dim=time_steps,
        n_scales=n_scales,
        feature_dim=feature_dim,
        embed_dim=embed_dim,
        eeg_attn_heads=int(model_config.get("eeg_attn_heads", 8)),
        eeg_attn_depth=int(model_config.get("eeg_attn_depth", 2)),
        visual_encoder_type=model_type,
        visual_dropout=float(model_config.get("visual_dropout", 0.1)),
    ).to(device)
    model.load_state_dict(ckpt["model"])

    feature_cache_path = args.feature_cache
    if feature_cache_path is None:
        cfg = ckpt.get("config", {})
        feat_cfg = cfg.get("features", {})
        feature_cache_path = feat_cfg.get("feature_cache", "cache/features_vitl_real.pt")

    cache = load_feature_cache(feature_cache_path)

    concept_to_label = None
    if args.split == "test":
        train_all = load_eeg_dataset(args.data_dir, "train", avg_trials=True, image_root="auto")
        concept_to_label = {c: int(l) for c, l in zip(train_all.concepts, train_all.labels.tolist())}

    records = load_eeg_dataset(
        args.data_dir, args.split, avg_trials=True,
        concept_to_label=concept_to_label, image_root="auto",
    )

    dataset = MultiScaleBlurDataset(records, cache, feature_keys, augment=False)
    model.eval()

    all_eeg_emb = []
    all_vis_emb = []
    aug = EEGTrainTransform(noise_std=0.01, channel_dropout_p=0.1, temporal_jitter=0, time_mask_frac=0.1)

    for eeg, scale_feats, _label in DataLoader(dataset, batch_size=args.batch_size, shuffle=False):
        eeg = eeg.to(device)
        scale_feats = scale_feats.to(device)
        with torch.no_grad():
            if args.tta > 0:
                preds_eeg = []
                for _ in range(args.tta):
                    eeg_aug = aug(eeg.clone())
                    eeg_emb, vis_emb = model(eeg_aug, scale_feats)
                    preds_eeg.append(F.normalize(eeg_emb, dim=-1))
                eeg_emb = F.normalize(torch.stack(preds_eeg).mean(0), dim=-1)
                vis_emb = F.normalize(vis_emb, dim=-1)
            else:
                eeg_emb, vis_emb = model(eeg, scale_feats)
                eeg_emb = F.normalize(eeg_emb, dim=-1)
                vis_emb = F.normalize(vis_emb, dim=-1)
        all_eeg_emb.append(eeg_emb.cpu())
        all_vis_emb.append(vis_emb.cpu())

    all_eeg_emb = torch.cat(all_eeg_emb, dim=0)
    all_vis_emb = torch.cat(all_vis_emb, dim=0)
    logits = all_eeg_emb @ all_vis_emb.T
    metrics = compute_retrieval_metrics(logits)

    result = {
        "logits": logits,
        "image_ids": records.image_ids,
        "concepts": records.concepts,
        "metrics": metrics,
        "split": args.split,
        "tta": args.tta,
        "feature_keys": feature_keys,
        "checkpoint": str(args.checkpoint),
    }

    output_path = args.output
    if output_path is None:
        tta_suffix = f"_tta{args.tta}" if args.tta > 0 else "_tta0"
        seed_str = Path(args.checkpoint).parent.name
        import re
        seed_match = re.search(r"seed(\d+)", seed_str)
        seed = int(seed_match.group(1)) if seed_match else 0
        model_tag = ckpt.get("model_type", "multiscale_blur")
        output_path = f"results/deep_{model_tag}_seed{seed}_{args.split}{tta_suffix}.logits.pt"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    print(f"Saved logits to {output_path}", flush=True)
    print(f"Metrics: {metrics}", flush=True)


if __name__ == "__main__":
    main()