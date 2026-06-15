#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brain2image.data import load_eeg_split
from brain2image.features import build_image_features, l2_normalize
from brain2image.io import ensure_dir, write_csv, write_json
from brain2image.metrics import compute_retrieval_metrics, ranks_from_logits
from brain2image.model import build_eeg_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval inference and export rankings.")
    parser.add_argument("--data-dir", default="image-eeg-data")
    parser.add_argument("--checkpoint", default="outputs/baseline/retrieval_baseline.pt")
    parser.add_argument("--output-dir", default="outputs/baseline")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--image-root", action="append", default=None)
    parser.add_argument("--selected-channel", action="append", default=None)
    parser.add_argument("--channel-jsonl", default=None)
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--cache-features", action="store_true")
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


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    device = choose_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    split = load_eeg_split(
        args.data_dir,
        args.split,
        avg_trials=True,
        selected_channels=args.selected_channel,
        eeg_channel_jsonl=args.channel_jsonl,
        image_roots=args.image_root,
    )
    if split.eeg.shape[1] != int(ckpt["channels"]):
        raise ValueError(
            f"Checkpoint expects {ckpt['channels']} EEG channels, but loaded {split.eeg.shape[1]}. "
            "Use the same --selected-channel arguments as training."
        )
    missing_images = sum(path is None for path in split.image_paths)
    if missing_images and str(ckpt["feature_backend"]) == "simple":
        print(
            f"Warning: {missing_images}/{len(split.image_paths)} {args.split} images were not found. "
            "Using deterministic image_id hash features for those candidates."
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

    candidate_features = build_image_features(
        split.image_ids,
        split.image_paths,
        embedding_dim=int(ckpt["embedding_dim"]),
        backend=str(ckpt["feature_backend"]),
        image_size=int(ckpt["image_size"]),
        device=device,
        batch_size=args.feature_batch_size,
        cache_path=(output_dir / f"{args.split}_features_{ckpt['feature_backend']}.pt") if args.cache_features else None,
    )
    with torch.no_grad():
        brain_features = l2_normalize(model(split.eeg.to(device)))
        logits = brain_features @ candidate_features.T

    ranks = ranks_from_logits(logits).cpu()
    topk = min(args.topk, ranks.shape[1])
    rows = []
    for query_idx, query_image_id in enumerate(split.image_ids):
        for rank_pos in range(topk):
            cand_idx = int(ranks[query_idx, rank_pos])
            rows.append(
                {
                    "query_index": query_idx,
                    "query_image_id": query_image_id,
                    "rank": rank_pos + 1,
                    "pred_image_id": split.image_ids[cand_idx],
                    "score": float(logits[query_idx, cand_idx].cpu().item()),
                }
            )

    csv_path = output_dir / f"retrieval_{args.split}_rankings.csv"
    json_path = output_dir / f"retrieval_{args.split}_rankings.json"
    metrics_path = output_dir / f"retrieval_{args.split}_metrics.json"
    write_csv(csv_path, rows, ["query_index", "query_image_id", "rank", "pred_image_id", "score"])
    write_json(
        json_path,
        [
            {
                "query_index": i,
                "query_image_id": split.image_ids[i],
                "ranked_image_ids": [split.image_ids[int(j)] for j in ranks[i, :topk]],
            }
            for i in range(ranks.shape[0])
        ],
    )
    if logits.shape[0] == logits.shape[1]:
        write_json(metrics_path, compute_retrieval_metrics(logits))

    torch.save({"logits": logits.cpu(), "image_ids": split.image_ids}, output_dir / f"retrieval_{args.split}_logits.pt")
    print(f"Wrote retrieval rankings: {csv_path}")
    print(f"Wrote retrieval JSON: {json_path}")
    if metrics_path.exists():
        print(f"Wrote retrieval metrics: {metrics_path}")


if __name__ == "__main__":
    main()
