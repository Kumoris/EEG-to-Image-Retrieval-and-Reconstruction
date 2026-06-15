from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from .data import concept_from_image_id, load_eeg_dataset, train_eeg_stats
from .features import features_for_ids, load_feature_cache
from .image_resolver import load_rgb
from .models import build_model_from_checkpoint
from .utils import choose_device, ensure_dir, l2_normalize, safe_torch_load, write_csv, write_json

LEAKAGE_POLICY = "uses only train images or deterministic placeholders as reconstruction sources; never copies test ground truth images"


def placeholder(path: Path, key: str, size: int) -> None:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    img = Image.new("RGB", (size, size), tuple(int(v) for v in digest[:3]))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), key[:28], fill=(255, 255, 255))
    img.save(path)


def write_image(src: Path | None, dst: Path, key: str, size: int) -> str:
    if src is not None and src.exists():
        img = load_rgb(src).resize((size, size), Image.BICUBIC)
        img.save(dst)
        return "train_nearest"
    placeholder(dst, key, size)
    return "prompt_placeholder"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="image-eeg-data")
    p.add_argument("--feature-cache", default="cache/features_rn50.pt")
    p.add_argument("--ckpt", default="runs/seed0/best.pt")
    p.add_argument("--retrieval-logits", default=None, help="Final retrieval logits .pt for ATM-S ensemble reconstruction.")
    p.add_argument("--output-dir", default="recons/seed0")
    p.add_argument(
        "--mode",
        default=None,
        choices=["checkpoint_train_nearest", "atms_ensemble_train_nearest"],
        help="Reconstruction backend. Defaults to checkpoint_train_nearest unless --retrieval-logits is supplied.",
    )
    p.add_argument("--method", default="auto", choices=["auto", "train_nearest", "diffusion", "img2img_topk"])
    p.add_argument("--diffusion-model", default=None, help="Reserved local diffusion model path; falls back when absent.")
    p.add_argument("--feature-key", default="image_clean_feature")
    p.add_argument("--topk", type=int, default=5, help="Number of predicted candidates to blend before train-nearest lookup.")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def _maybe_note_diffusion_fallback(args: argparse.Namespace) -> str | None:
    if args.method not in {"auto", "diffusion", "img2img_topk"}:
        return None
    if args.diffusion_model and Path(args.diffusion_model).exists():
        return "local diffusion model path was provided, but diffusion reconstruction is not implemented in this lightweight wrapper; used train_nearest fallback"
    if args.diffusion_model:
        return f"diffusion model path does not exist: {args.diffusion_model}; used train_nearest fallback"
    try:
        import diffusers  # noqa: F401

        return "diffusers is installed, but no local diffusion model path was provided; used train_nearest fallback"
    except Exception as exc:
        return f"diffusers unavailable; used train_nearest fallback: {exc}"


def _load_train_records(data_dir: str | Path):
    return load_eeg_dataset(data_dir, "train", avg_trials=True, image_root="auto")


def _write_rows(out_dir: Path, rows: list[dict], summary: dict) -> None:
    fields = [
        "query_index",
        "query_image_id_metadata",
        "predicted_candidate_index",
        "predicted_candidate_image_id",
        "predicted_candidate_concept",
        "topk_candidate_image_ids",
        "nearest_train_index",
        "nearest_train_image_id",
        "nearest_train_concept",
        "score",
        "output",
        "source",
        "source_kind",
        "leakage_policy",
    ]
    write_csv(out_dir / "manifest.csv", rows, fields)
    write_json(out_dir / "summary.json", summary)
    print(f"Wrote {len(rows)} PNG reconstructions to {out_dir}", flush=True)


@torch.no_grad()
def reconstruct_from_ensemble_logits(args: argparse.Namespace) -> None:
    if not args.retrieval_logits:
        raise ValueError("--retrieval-logits is required for --mode atms_ensemble_train_nearest")

    out_dir = ensure_dir(args.output_dir)
    cache = load_feature_cache(args.feature_cache)
    train = _load_train_records(args.data_dir)
    test = load_eeg_dataset(args.data_dir, "test", avg_trials=True, image_root="auto")
    obj = safe_torch_load(args.retrieval_logits, map_location="cpu")
    if "logits" not in obj:
        raise KeyError(f"{args.retrieval_logits} does not contain a 'logits' tensor")

    retrieval_logits = obj["logits"].float()
    candidate_ids = [str(x) for x in obj.get("image_ids") or cache["image_ids"][cache["split_ranges"]["test"][0] : cache["split_ranges"]["test"][1]]]
    if retrieval_logits.shape[1] != len(candidate_ids):
        raise ValueError(f"logits columns ({retrieval_logits.shape[1]}) do not match candidate ids ({len(candidate_ids)})")
    if retrieval_logits.shape[0] > len(test.image_ids):
        raise ValueError(f"logits rows ({retrieval_logits.shape[0]}) exceed test metadata rows ({len(test.image_ids)})")

    train_feats = features_for_ids(cache, train.image_ids, args.feature_key)
    candidate_feats = features_for_ids(cache, candidate_ids, args.feature_key)
    topk = min(max(1, args.topk), retrieval_logits.shape[1])
    ranks = retrieval_logits.topk(k=topk, dim=1)
    weights = torch.softmax(ranks.values, dim=1)
    query_proxy = l2_normalize((candidate_feats[ranks.indices] * weights.unsqueeze(-1)).sum(dim=1))
    train_logits = query_proxy @ train_feats.T
    nearest = train_logits.argmax(dim=1)
    diffusion_note = _maybe_note_diffusion_fallback(args)
    if diffusion_note:
        print(diffusion_note, flush=True)

    rows = []
    for i in range(retrieval_logits.shape[0]):
        train_idx = int(nearest[i].item())
        pred_idx = int(ranks.indices[i, 0].item())
        pred_id = candidate_ids[pred_idx]
        dst = out_dir / f"{i:03d}.png"
        source_kind = write_image(train.image_paths[train_idx], dst, train.image_ids[train_idx], args.image_size)
        rows.append(
            {
                "query_index": i,
                "query_image_id_metadata": test.image_ids[i],
                "predicted_candidate_index": pred_idx,
                "predicted_candidate_image_id": pred_id,
                "predicted_candidate_concept": concept_from_image_id(pred_id),
                "topk_candidate_image_ids": ";".join(candidate_ids[int(j)] for j in ranks.indices[i].tolist()),
                "nearest_train_index": train_idx,
                "nearest_train_image_id": train.image_ids[train_idx],
                "nearest_train_concept": train.concepts[train_idx],
                "score": float(train_logits[i, train_idx].item()),
                "output": str(dst),
                "source": source_kind,
                "source_kind": source_kind,
                "leakage_policy": LEAKAGE_POLICY,
            }
        )

    summary = {
        "num_images": len(rows),
        "method": "atms_ensemble_train_nearest",
        "requested_method": args.method,
        "retrieval_logits": str(args.retrieval_logits),
        "feature_cache": str(args.feature_cache),
        "feature_key": args.feature_key,
        "topk": topk,
        "image_size": args.image_size,
        "reconstruction_dir": str(out_dir),
        "manifest": str(out_dir / "manifest.csv"),
        "diffusion_note": diffusion_note,
        "retrieval_metrics": obj.get("metrics"),
        "retrieval_weights": obj.get("weights"),
        "leakage_policy": LEAKAGE_POLICY,
        "source_counts": {
            "train_nearest": sum(1 for row in rows if row["source_kind"] == "train_nearest"),
            "prompt_placeholder": sum(1 for row in rows if row["source_kind"] == "prompt_placeholder"),
        },
    }
    _write_rows(out_dir, rows, summary)


@torch.no_grad()
def reconstruct_from_checkpoint(args: argparse.Namespace) -> None:
    out_dir = ensure_dir(args.output_dir)
    device = choose_device(args.device)
    ckpt = safe_torch_load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("config", {})
    cache = load_feature_cache(args.feature_cache)
    train_ref = load_eeg_dataset(args.data_dir, "train", avg_trials=bool(cfg.get("data", {}).get("avg_trials_train", False)), selected_channels=cfg.get("data", {}).get("selected_channels"), image_root="auto")
    mean, std = train_eeg_stats(train_ref)
    train = load_eeg_dataset(args.data_dir, "train", avg_trials=True, selected_channels=cfg.get("data", {}).get("selected_channels"), image_root="auto").normalize(mean, std)
    test = load_eeg_dataset(args.data_dir, "test", avg_trials=True, selected_channels=cfg.get("data", {}).get("selected_channels"), image_root="auto").normalize(mean, std)
    model = build_model_from_checkpoint(ckpt, device).eval()
    train_feats = features_for_ids(cache, train.image_ids, "image_clean_feature").to(device)
    test_loader = torch.utils.data.DataLoader(test.eeg, batch_size=512, shuffle=False)
    eeg_feats = []
    for eeg in test_loader:
        out = model(eeg.to(device))
        eeg_feats.append(l2_normalize(0.6 * out["fusion"] + 0.4 * out["aligned"]["fusion"]).cpu())
    eeg_feats = torch.cat(eeg_feats, dim=0).to(device)
    logits = eeg_feats @ train_feats.T
    nearest = logits.argmax(dim=1).cpu().tolist()
    rows = []
    diffusion_note = _maybe_note_diffusion_fallback(args)
    if diffusion_note:
        print(diffusion_note, flush=True)
    for i, train_idx in enumerate(nearest):
        dst = out_dir / f"{i:03d}.png"
        kind = write_image(train.image_paths[train_idx], dst, train.image_ids[train_idx], args.image_size)
        rows.append(
            {
                "query_index": i,
                "query_image_id_metadata": test.image_ids[i],
                "predicted_candidate_index": "",
                "predicted_candidate_image_id": "",
                "predicted_candidate_concept": "",
                "topk_candidate_image_ids": "",
                "nearest_train_index": train_idx,
                "nearest_train_image_id": train.image_ids[train_idx],
                "nearest_train_concept": train.concepts[train_idx],
                "score": float(logits[i, train_idx].cpu().item()),
                "output": str(dst),
                "source": kind,
                "source_kind": kind,
                "leakage_policy": LEAKAGE_POLICY,
            }
        )
    summary = {
        "num_images": len(rows),
        "method": "checkpoint_train_nearest",
        "requested_method": args.method,
        "diffusion_note": diffusion_note,
        "leakage_policy": LEAKAGE_POLICY,
    }
    _write_rows(out_dir, rows, summary)


def main() -> None:
    args = parse_args()
    mode = args.mode or ("atms_ensemble_train_nearest" if args.retrieval_logits else "checkpoint_train_nearest")
    if mode == "atms_ensemble_train_nearest":
        reconstruct_from_ensemble_logits(args)
    else:
        reconstruct_from_checkpoint(args)


if __name__ == "__main__":
    main()
