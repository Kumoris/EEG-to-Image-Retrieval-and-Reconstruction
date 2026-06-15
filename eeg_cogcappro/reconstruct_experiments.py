from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import ImageEnhance

from .data import concept_from_image_id, load_eeg_dataset
from .features import features_for_ids, load_feature_cache
from .image_resolver import load_rgb
from .reconstruct import LEAKAGE_POLICY, placeholder
from .utils import ensure_dir, l2_normalize, safe_torch_load, write_csv, write_json


METHODS = [
    "train_nearest_top1",
    "train_nearest_rerank_topk",
    "concept_train_nearest",
    "postprocess_sharp_color",
    "diffusion_prompt",
    "diffusion_img2img_train_source",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate legal reconstruction candidates for final ATM-S ensemble.")
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--data-dir", default="image-eeg-data")
    p.add_argument("--feature-cache", default="cache/features_vitl.pt")
    p.add_argument("--retrieval-logits", default="results/multi_encoder_ensemble/retrieval_test_logits.pt")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--feature-key", default="image_clean_feature")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--train-candidates", type=int, default=25)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--diffusion-model", default=None)
    p.add_argument("--diffusion-steps", type=int, default=1)
    p.add_argument("--guidance-scale", type=float, default=0.0)
    p.add_argument("--strength", type=float, default=0.55)
    p.add_argument("--seed", type=int, default=20260427)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def _load_common(args: argparse.Namespace):
    cache = load_feature_cache(args.feature_cache)
    train = load_eeg_dataset(args.data_dir, "train", avg_trials=True, image_root="auto")
    test = load_eeg_dataset(args.data_dir, "test", avg_trials=True, image_root="auto")
    obj = safe_torch_load(args.retrieval_logits, map_location="cpu")
    logits = obj["logits"].float()
    a, b = cache["split_ranges"]["test"]
    candidate_ids = [str(x) for x in obj.get("image_ids") or cache["image_ids"][int(a) : int(b)]]
    train_feats = features_for_ids(cache, train.image_ids, args.feature_key)
    candidate_feats = features_for_ids(cache, candidate_ids, args.feature_key)
    topk = min(max(1, args.topk), logits.shape[1])
    ranks = logits.topk(k=topk, dim=1)
    return cache, train, test, obj, logits, candidate_ids, train_feats, candidate_feats, ranks


def _write_image(src: Path | None, dst: Path, key: str, size: int, *, postprocess: bool = False) -> str:
    if src is None or not src.exists():
        placeholder(dst, key, size)
        return "prompt_placeholder"
    img = load_rgb(src).resize((size, size))
    if postprocess:
        img = ImageEnhance.Contrast(img).enhance(1.08)
        img = ImageEnhance.Color(img).enhance(1.06)
        img = ImageEnhance.Sharpness(img).enhance(1.18)
    img.save(dst)
    return "train_nearest_postprocessed" if postprocess else "train_nearest"


def _select_indices(
    method: str,
    train,
    logits: torch.Tensor,
    candidate_ids: list[str],
    train_feats: torch.Tensor,
    candidate_feats: torch.Tensor,
    ranks,
    *,
    train_candidates: int,
) -> tuple[list[int], list[float], list[str]]:
    if method in {"diffusion_prompt", "diffusion_img2img_train_source"}:
        raise ValueError("Diffusion methods are handled separately.")
    if method in {"train_nearest_top1", "concept_train_nearest", "postprocess_sharp_color"}:
        query = candidate_feats[ranks.indices[:, 0]]
    else:
        weights = torch.softmax(ranks.values, dim=1)
        query = l2_normalize((candidate_feats[ranks.indices] * weights.unsqueeze(-1)).sum(dim=1))

    sims = query @ train_feats.T
    selected: list[int] = []
    scores: list[float] = []
    notes: list[str] = []
    for i in range(logits.shape[0]):
        if method == "concept_train_nearest":
            concept = concept_from_image_id(candidate_ids[int(ranks.indices[i, 0])])
            pool = [j for j, c in enumerate(train.concepts) if c == concept]
            if pool:
                pool_scores = sims[i, pool]
                local = int(pool_scores.argmax().item())
                idx = int(pool[local])
                selected.append(idx)
                scores.append(float(sims[i, idx].item()))
                notes.append(f"restricted_to_predicted_concept:{concept}")
                continue
            notes.append(f"predicted_concept_missing_in_train:{concept}")

        if method == "train_nearest_rerank_topk":
            n = min(train_candidates, sims.shape[1])
            train_vals, train_idx = sims[i].topk(k=n)
            cand = candidate_feats[ranks.indices[i]]
            train_to_candidate = (train_feats[train_idx] @ cand.T).max(dim=1).values
            retrieval_prior = torch.softmax(ranks.values[i], dim=0).max().expand_as(train_to_candidate)
            rerank = 0.65 * train_vals + 0.25 * train_to_candidate + 0.10 * retrieval_prior
            best = int(rerank.argmax().item())
            idx = int(train_idx[best].item())
            selected.append(idx)
            scores.append(float(rerank[best].item()))
            notes.append("rerank=0.65*query_train+0.25*train_candidate+0.10*retrieval_prior")
            continue

        idx = int(sims[i].argmax().item())
        selected.append(idx)
        scores.append(float(sims[i, idx].item()))
        notes.append("global_train_nearest")
    return selected, scores, notes


def _write_rows(args: argparse.Namespace, train, test, logits, candidate_ids, ranks, selected, scores, notes, *, postprocess: bool) -> None:
    out_dir = ensure_dir(args.output_dir)
    rows = []
    for i, train_idx in enumerate(selected):
        pred_idx = int(ranks.indices[i, 0].item())
        pred_id = candidate_ids[pred_idx]
        dst = out_dir / f"{i:03d}.png"
        source = _write_image(train.image_paths[train_idx], dst, train.image_ids[train_idx], args.image_size, postprocess=postprocess)
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
                "score": scores[i],
                "output": str(dst),
                "source": source,
                "source_kind": source,
                "selection_note": notes[i],
                "leakage_policy": LEAKAGE_POLICY,
            }
        )
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
        "selection_note",
        "leakage_policy",
    ]
    write_csv(out_dir / "manifest.csv", rows, fields)
    write_json(
        out_dir / "summary.json",
        {
            "method": args.method,
            "num_images": len(rows),
            "retrieval_logits": args.retrieval_logits,
            "feature_cache": args.feature_cache,
            "feature_key": args.feature_key,
            "topk": args.topk,
            "train_candidates": args.train_candidates,
            "image_size": args.image_size,
            "leakage_policy": LEAKAGE_POLICY,
            "source_counts": {kind: sum(1 for row in rows if row["source_kind"] == kind) for kind in sorted({row["source_kind"] for row in rows})},
        },
    )
    print(f"Wrote {len(rows)} reconstructions to {out_dir}")


def _write_diffusion_skip(args: argparse.Namespace, reason: str) -> None:
    out_dir = ensure_dir(args.output_dir)
    write_json(
        out_dir / "summary.json",
        {
            "method": args.method,
            "status": "skipped",
            "skip_reason": reason,
            "num_images": 0,
            "diffusion_model": args.diffusion_model,
            "leakage_policy": LEAKAGE_POLICY,
        },
    )
    print(f"Skipped {args.method}: {reason}")


def _prompt_for(candidate_ids: list[str], ranks, i: int) -> str:
    concepts = []
    for idx in ranks.indices[i].tolist()[:3]:
        concept = concept_from_image_id(candidate_ids[int(idx)]).replace("_", " ")
        if concept not in concepts:
            concepts.append(concept)
    joined = ", ".join(concepts)
    return f"a centered high quality photo of {joined}, simple background, natural color, sharp object"


def _load_text2image_pipeline(args: argparse.Namespace):
    from diffusers import AutoPipelineForText2Image

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(
        args.diffusion_model,
        torch_dtype=dtype,
        variant="fp16",
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe, device


def _load_img2img_pipeline(args: argparse.Namespace):
    from diffusers import AutoPipelineForImage2Image

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = AutoPipelineForImage2Image.from_pretrained(
        args.diffusion_model,
        torch_dtype=dtype,
        variant="fp16",
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe, device


def _run_diffusion(args: argparse.Namespace) -> None:
    if not args.diffusion_model:
        _write_diffusion_skip(args, "No local diffusion model id/path was provided.")
        return
    out_dir = ensure_dir(args.output_dir)
    _, train, test, _, logits, candidate_ids, train_feats, candidate_feats, ranks = _load_common(args)
    selected, scores, notes = _select_indices(
        "train_nearest_top1",
        train,
        logits,
        candidate_ids,
        train_feats,
        candidate_feats,
        ranks,
        train_candidates=args.train_candidates,
    )
    try:
        if args.method == "diffusion_prompt":
            pipe, device = _load_text2image_pipeline(args)
        else:
            pipe, device = _load_img2img_pipeline(args)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _write_diffusion_skip(args, f"Could not load diffusion pipeline: {exc}")
        return

    rows = []
    for i, train_idx in enumerate(selected):
        prompt = _prompt_for(candidate_ids, ranks, i)
        gen = torch.Generator(device=device).manual_seed(args.seed + i)
        pred_idx = int(ranks.indices[i, 0].item())
        pred_id = candidate_ids[pred_idx]
        dst = out_dir / f"{i:03d}.png"
        kwargs = {
            "prompt": prompt,
            "num_inference_steps": args.diffusion_steps,
            "guidance_scale": args.guidance_scale,
            "generator": gen,
            "height": 512,
            "width": 512,
        }
        if args.method == "diffusion_img2img_train_source":
            src = train.image_paths[train_idx]
            if src is None or not src.exists():
                placeholder(dst, train.image_ids[train_idx], args.image_size)
                source = "prompt_placeholder"
            else:
                init_image = load_rgb(src).resize((512, 512))
                img = pipe(image=init_image, strength=args.strength, **kwargs).images[0].resize((args.image_size, args.image_size))
                img.save(dst)
                source = "generated_diffusion_img2img_train_source"
        else:
            img = pipe(**kwargs).images[0].resize((args.image_size, args.image_size))
            img.save(dst)
            source = "generated_diffusion_prompt"
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
                "score": scores[i],
                "output": str(dst),
                "source": source,
                "source_kind": source,
                "selection_note": notes[i],
                "prompt": prompt,
                "leakage_policy": LEAKAGE_POLICY,
            }
        )
        if (i + 1) % 25 == 0:
            print(f"{args.method}: generated {i + 1}/{len(selected)}", flush=True)

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
        "selection_note",
        "prompt",
        "leakage_policy",
    ]
    write_csv(out_dir / "manifest.csv", rows, fields)
    write_json(
        out_dir / "summary.json",
        {
            "method": args.method,
            "num_images": len(rows),
            "diffusion_model": args.diffusion_model,
            "diffusion_variant": "fp16",
            "diffusion_steps": args.diffusion_steps,
            "guidance_scale": args.guidance_scale,
            "strength": args.strength if args.method == "diffusion_img2img_train_source" else None,
            "seed": args.seed,
            "image_size": args.image_size,
            "leakage_policy": LEAKAGE_POLICY,
            "source_counts": {kind: sum(1 for row in rows if row["source_kind"] == kind) for kind in sorted({row["source_kind"] for row in rows})},
        },
    )
    print(f"Wrote {len(rows)} diffusion reconstructions to {out_dir}")


def main() -> None:
    args = parse_args()
    if args.method in {"diffusion_prompt", "diffusion_img2img_train_source"}:
        _run_diffusion(args)
        return

    _, train, test, _, logits, candidate_ids, train_feats, candidate_feats, ranks = _load_common(args)
    selected, scores, notes = _select_indices(
        args.method,
        train,
        logits,
        candidate_ids,
        train_feats,
        candidate_feats,
        ranks,
        train_candidates=args.train_candidates,
    )
    _write_rows(args, train, test, logits, candidate_ids, ranks, selected, scores, notes, postprocess=args.method == "postprocess_sharp_color")


if __name__ == "__main__":
    main()
