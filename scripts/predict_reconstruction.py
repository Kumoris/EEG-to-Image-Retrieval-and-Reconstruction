#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brain2image.data import load_eeg_split
from brain2image.features import l2_normalize
from brain2image.io import ensure_dir, write_csv, write_json
from brain2image.model import build_eeg_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nearest-neighbor reconstruction baseline.")
    parser.add_argument("--data-dir", default="image-eeg-data")
    parser.add_argument("--checkpoint", default="outputs/baseline/retrieval_baseline.pt")
    parser.add_argument("--output-dir", default="outputs/baseline")
    parser.add_argument("--image-root", action="append", default=None)
    parser.add_argument("--selected-channel", action="append", default=None)
    parser.add_argument("--channel-jsonl", default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text[:80] or "sample"


def placeholder_image(path: Path, key: str, size: int) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ImportError("Pillow is required to write reconstruction images.") from exc

    digest = hashlib.sha256(key.encode("utf-8")).digest()
    color = tuple(int(x) for x in digest[:3])
    img = Image.new("RGB", (size, size), color=color)
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), key[:24], fill=(255, 255, 255))
    img.save(path)


def copy_or_placeholder(src: Path | None, dst: Path, key: str, size: int) -> str:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to write reconstruction images.") from exc

    if src is not None and src.exists():
        with Image.open(src) as img:
            img = img.convert("RGB").resize((size, size))
            img.save(dst)
        return "nearest_image"
    placeholder_image(dst, key, size)
    return "placeholder_missing_image"


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    recon_dir = ensure_dir(output_dir / "reconstructions")
    device = choose_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    train_split = load_eeg_split(
        args.data_dir,
        "train",
        avg_trials=True,
        selected_channels=args.selected_channel,
        eeg_channel_jsonl=args.channel_jsonl,
        image_roots=args.image_root,
    )
    test_split = load_eeg_split(
        args.data_dir,
        "test",
        avg_trials=True,
        selected_channels=args.selected_channel,
        eeg_channel_jsonl=args.channel_jsonl,
        image_roots=args.image_root,
    )
    if test_split.eeg.shape[1] != int(ckpt["channels"]):
        raise ValueError(
            f"Checkpoint expects {ckpt['channels']} EEG channels, but loaded {test_split.eeg.shape[1]}. "
            "Use the same --selected-channel arguments as training."
        )
    missing_train_images = sum(path is None for path in train_split.image_paths)
    if missing_train_images:
        print(
            f"Warning: {missing_train_images}/{len(train_split.image_paths)} train images were not found. "
            "Missing nearest-neighbor images will be replaced by placeholder PNGs."
        )

    model = build_eeg_encoder(
        str(ckpt.get("model_kind", "conv")),
        channels=int(ckpt["channels"]),
        embedding_dim=int(ckpt["embedding_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        dropout=float(ckpt.get("dropout", 0.1)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    train_features = l2_normalize(ckpt["train_image_features"].to(device))

    with torch.no_grad():
        brain_features = l2_normalize(model(test_split.eeg.to(device)))
        logits = brain_features @ train_features.T
        nearest = logits.argmax(dim=1).cpu()

    rows = []
    for query_idx, train_idx_tensor in enumerate(nearest):
        train_idx = int(train_idx_tensor)
        query_id = test_split.image_ids[query_idx]
        nearest_id = train_split.image_ids[train_idx]
        filename = f"{query_idx:04d}_{safe_name(query_id)}.png"
        out_path = recon_dir / filename
        source_kind = copy_or_placeholder(
            train_split.image_paths[train_idx],
            out_path,
            nearest_id,
            args.image_size,
        )
        rows.append(
            {
                "query_index": query_idx,
                "query_image_id": query_id,
                "nearest_train_index": train_idx,
                "nearest_image_id": nearest_id,
                "score": float(logits[query_idx, train_idx].cpu().item()),
                "output_path": str(out_path),
                "source_kind": source_kind,
            }
        )

    manifest = output_dir / "reconstruction_manifest.csv"
    write_csv(
        manifest,
        rows,
        [
            "query_index",
            "query_image_id",
            "nearest_train_index",
            "nearest_image_id",
            "score",
            "output_path",
            "source_kind",
        ],
    )
    write_json(
        output_dir / "reconstruction_summary.json",
        {
            "num_reconstructions": len(rows),
            "reconstruction_dir": str(recon_dir),
            "manifest": str(manifest),
            "placeholder_count": sum(row["source_kind"] == "placeholder_missing_image" for row in rows),
        },
    )
    print(f"Wrote reconstructions: {recon_dir}")
    print(f"Wrote reconstruction manifest: {manifest}")


if __name__ == "__main__":
    main()
