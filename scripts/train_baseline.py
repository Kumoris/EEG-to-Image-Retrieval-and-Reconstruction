#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brain2image.data import load_eeg_split
from brain2image.features import build_image_features
from brain2image.io import ensure_dir, write_json
from brain2image.model import build_eeg_encoder
from brain2image.seed import set_seed
from brain2image.train import train_retrieval_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal EEG-to-image retrieval baseline.")
    parser.add_argument("--data-dir", default="image-eeg-data", help="Directory containing train.pt/test.pt.")
    parser.add_argument("--output-dir", default="outputs/baseline", help="Directory for checkpoints and logs.")
    parser.add_argument("--image-root", action="append", default=None, help="Optional extra image root. Can be repeated.")
    parser.add_argument("--selected-channel", action="append", default=None, help="Optional EEG channel name. Can be repeated.")
    parser.add_argument("--channel-jsonl", default=None, help="Optional EEG channel JSONL path.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--feature-backend",
        choices=[
            "simple",
            "hash",
            "torchvision-rn18",
            "torchvision-rn50",
            "torchvision-rn18-logits",
            "torchvision-rn50-logits",
        ],
        default="simple",
    )
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--cache-features", action="store_true")
    parser.add_argument("--model-kind", choices=["conv", "tsconv"], default="conv")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--scm-weight", type=float, default=0.0)
    parser.add_argument("--scm-topk", type=int, default=10)
    parser.add_argument("--init-checkpoint", default=None, help="Optional checkpoint to initialize model weights.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--quiet", action="store_true", help="Disable per-epoch terminal progress.")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    print(f"Using device: {device}", flush=True)
    print(f"Loading train split from: {args.data_dir}", flush=True)

    split = load_eeg_split(
        args.data_dir,
        "train",
        avg_trials=True,
        selected_channels=args.selected_channel,
        eeg_channel_jsonl=args.channel_jsonl,
        image_roots=args.image_root,
    )
    missing_images = sum(path is None for path in split.image_paths)
    if missing_images and args.feature_backend == "simple":
        print(
            f"Warning: {missing_images}/{len(split.image_paths)} train images were not found. "
            "Using deterministic image_id hash features for those samples."
        )
    image_features = build_image_features(
        split.image_ids,
        split.image_paths,
        embedding_dim=args.embedding_dim,
        backend=args.feature_backend,
        image_size=args.image_size,
        device="cpu",
        batch_size=args.feature_batch_size,
        cache_path=(output_dir / f"train_features_{args.feature_backend}.pt") if args.cache_features else None,
    )
    if image_features.shape[1] != args.embedding_dim:
        print(
            f"Adjusting embedding_dim from {args.embedding_dim} to image feature dim {image_features.shape[1]}",
            flush=True,
        )
        args.embedding_dim = int(image_features.shape[1])

    channels, time_steps = split.eeg.shape[1], split.eeg.shape[2]
    print(
        f"Loaded {len(split.image_ids)} samples | EEG shape={tuple(split.eeg.shape)} | "
        f"image feature dim={args.embedding_dim}",
        flush=True,
    )
    print(
        f"Training for {args.epochs} epochs | batch_size={args.batch_size} | lr={args.lr}",
        flush=True,
    )
    model = build_eeg_encoder(
        args.model_kind,
        channels=channels,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    if args.init_checkpoint:
        init = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if int(init["channels"]) != channels or int(init["embedding_dim"]) != args.embedding_dim:
            raise ValueError("Initial checkpoint shape does not match the current data/features.")
        model.load_state_dict(init["model_state"])
        print(f"Initialized model from: {args.init_checkpoint}", flush=True)
    history = train_retrieval_model(
        model,
        split.eeg,
        image_features,
        labels=torch.tensor(split.labels),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        device=device,
        scm_weight=args.scm_weight,
        scm_topk=args.scm_topk,
        verbose=not args.quiet,
    )

    checkpoint_path = output_dir / "retrieval_baseline.pt"
    torch.save(
        {
            "model_state": model.cpu().state_dict(),
            "channels": channels,
            "time_steps": time_steps,
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "model_kind": args.model_kind,
            "dropout": args.dropout,
            "feature_backend": args.feature_backend,
            "image_size": args.image_size,
            "feature_batch_size": args.feature_batch_size,
            "train_image_ids": split.image_ids,
            "train_labels": split.labels,
            "train_texts": split.texts,
            "train_raw_images": split.raw_images,
            "train_image_features": image_features.cpu(),
            "config": vars(args),
        },
        checkpoint_path,
    )
    write_json(output_dir / "train_history.json", history)
    write_json(
        output_dir / "train_summary.json",
        {
            "checkpoint": str(checkpoint_path),
            "num_train": len(split.image_ids),
            "eeg_shape": list(split.eeg.shape),
            "final_train_loss": history[-1]["train_loss"] if history else None,
            "device": str(device),
        },
    )
    print(f"Wrote checkpoint: {checkpoint_path}")
    print(f"Wrote train history: {output_dir / 'train_history.json'}")


if __name__ == "__main__":
    main()
