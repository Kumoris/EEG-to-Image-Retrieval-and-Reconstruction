from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import EEGDataset, deterministic_group_split, load_eeg_dataset, subset_records, train_eeg_stats
from .features import features_for_ids, load_feature_cache
from .losses import scm_loss, sth_align_loss, symmetric_clip_loss
from .models import CogCapProModel
from .transforms_eeg import EEGTrainTransform
from .utils import autocast_context, choose_device, compute_retrieval_metrics, ensure_dir, load_yaml, set_seed, write_json


FEATURE_KEYS = {
    "img": "image_clean_feature",
    "text": "text_feature",
    "depth": "depth_feature",
    "edge": "edge_feature",
    "fovea_low": "image_fovea_low",
    "fovea_mid": "image_fovea_mid",
    "fovea_high": "image_fovea_high",
}


def _targets(cache: dict, image_ids: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: features_for_ids(cache, image_ids, key).to(device) for name, key in FEATURE_KEYS.items() if key in cache}


def _batch_targets(all_targets: dict[str, torch.Tensor], idx: torch.Tensor) -> dict[str, torch.Tensor]:
    return {k: v[idx] for k, v in all_targets.items()}


def _um_target(batch_targets: dict[str, torch.Tensor], score_bank: torch.Tensor, idx: torch.Tensor, enabled: bool, z: float) -> tuple[torch.Tensor, dict[str, float]]:
    if not enabled or not {"fovea_low", "fovea_mid", "fovea_high"}.issubset(batch_targets):
        return batch_targets["img"], {"hard": 0.0, "normal": 1.0, "easy": 0.0}
    scores = score_bank[idx].to(batch_targets["img"].device)
    mean = score_bank.mean().to(scores.device)
    std = score_bank.std().clamp_min(1e-6).to(scores.device)
    hard = scores < mean - z * std
    easy = scores > mean + z * std
    out = batch_targets["fovea_mid"].clone()
    out[hard] = batch_targets["fovea_low"][hard]
    out[easy] = batch_targets["fovea_high"][easy]
    n = max(1, int(scores.numel()))
    return out, {"hard": float(hard.float().mean().item()), "easy": float(easy.float().mean().item()), "normal": float((~(hard | easy)).float().mean().item())}


@torch.no_grad()
def evaluate(model: CogCapProModel, records, cache: dict, device: torch.device, weights: dict[str, float]) -> dict[str, float]:
    model.eval()
    ds = EEGDataset(records)
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
    imgs = features_for_ids(cache, records.image_ids, "image_clean_feature").to(device)
    logits_parts = []
    for batch in loader:
        out = model(batch["eeg"].to(device))
        experts = out["experts"]
        aligned = out["aligned"]
        logits = weights.get("img", 0.3) * (experts["img"] @ imgs.T)
        logits = logits + weights.get("fusion", 0.5) * (out["fusion"] @ imgs.T)
        logits = logits + weights.get("aligned", 0.2) * (aligned["fusion"] @ imgs.T)
        logits_parts.append(logits.cpu())
    return compute_retrieval_metrics(torch.cat(logits_parts, dim=0))


def run_stage(
    model: CogCapProModel,
    train_records,
    val_records,
    cache: dict,
    cfg: dict,
    device: torch.device,
    output_dir: Path,
    *,
    stage: str,
    epochs: int,
    lr: float,
    freeze_experts: bool = False,
) -> tuple[float, list[dict]]:
    if freeze_experts:
        for p in model.experts.parameters():
            p.requires_grad = False
        for p in model.fusion.parameters():
            p.requires_grad = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=float(cfg["train"]["weight_decay"]))
    steps = max(1, epochs * max(1, len(train_records.eeg) // int(cfg["train"]["batch_size"])))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"]["amp"]) and device.type == "cuda")
    transform = EEGTrainTransform(**cfg.get("augment", {}))
    ds = EEGDataset(train_records, transform=transform)
    loader = DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=True, num_workers=0, drop_last=False)
    all_targets = _targets(cache, train_records.image_ids, device)
    score_bank = torch.zeros(len(train_records.eeg), dtype=torch.float32)
    best = -1.0
    hist: list[dict] = []
    accum = int(cfg["train"].get("grad_accum", 1))
    for epoch in range(1, epochs + 1):
        model.train()
        start = time.perf_counter()
        loss_sum = 0.0
        n_seen = 0
        um_acc = {"hard": 0.0, "normal": 0.0, "easy": 0.0}
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, start=1):
            eeg = batch["eeg"].to(device)
            local_idx = batch["index"]
            if not torch.equal(train_records.indices, torch.arange(len(train_records.indices))):
                lookup = {int(v): i for i, v in enumerate(train_records.indices.tolist())}
                idx = torch.tensor([lookup[int(v)] for v in local_idx.tolist()], dtype=torch.long)
            else:
                idx = local_idx.long()
            labels = train_records.labels[idx].to(device)
            bt = _batch_targets(all_targets, idx.to(device))
            with autocast_context(device, bool(cfg["train"]["amp"])):
                out = model(eeg)
                experts = out["experts"]
                aligned = out["aligned"]
                image_um, ratios = _um_target(bt, score_bank, idx.cpu(), bool(cfg["um"]["enabled"]) and stage == "A", float(cfg["um"]["z"]))
                logit_scale = out["logit_scale"]
                if stage == "B":
                    loss = sth_align_loss(
                        aligned,
                        {"img": bt["img"], "text": bt["text"], "depth": bt["depth"], "edge": bt["edge"], "fusion": bt["img"]},
                        lambda_mse=float(cfg["loss"]["lambda_mse"]),
                        lambda_cos=float(cfg["loss"]["lambda_cos"]),
                        lambda_reg=float(cfg["loss"]["lambda_reg"]),
                    )
                    loss = loss + symmetric_clip_loss(aligned["fusion"], bt["img"], logit_scale)
                else:
                    text_w = float(cfg["loss"]["w_text"]) if epoch <= 30 else min(0.1, float(cfg["loss"]["w_text"]))
                    loss = float(cfg["loss"]["w_img"]) * symmetric_clip_loss(experts["img"], image_um, logit_scale)
                    loss = loss + text_w * symmetric_clip_loss(experts["text"], bt["text"], logit_scale)
                    loss = loss + float(cfg["loss"]["w_depth"]) * symmetric_clip_loss(experts["depth"], bt["depth"], logit_scale)
                    loss = loss + float(cfg["loss"]["w_edge"]) * symmetric_clip_loss(experts["edge"], bt["edge"], logit_scale)
                    loss = loss + float(cfg["loss"]["w_fusion"]) * symmetric_clip_loss(out["fusion"], bt["img"], logit_scale)
                    loss = loss + float(cfg["loss"]["w_scm_fusion"]) * scm_loss(out["fusion"], bt["img"], labels, int(cfg["loss"]["scm_top_k"]), float(cfg["loss"]["tau"]))
                    loss = loss + float(cfg["loss"]["w_scm_img"]) * scm_loss(experts["img"], bt["img"], labels, int(cfg["loss"]["scm_top_k"]), float(cfg["loss"]["tau"]))
                loss = loss / accum
            scaler.scale(loss).backward()
            if step % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            with torch.no_grad():
                sim = torch.sum(out["fusion"].detach().float() * bt["img"].float(), dim=-1).cpu()
                score_bank[idx.cpu()] = float(cfg["um"]["gamma"]) * sim + (1.0 - float(cfg["um"]["gamma"])) * score_bank[idx.cpu()]
            bs = eeg.shape[0]
            loss_sum += float(loss.detach().cpu().item()) * bs * accum
            n_seen += bs
            for k in um_acc:
                um_acc[k] += ratios[k] * bs
        metrics = evaluate(model, val_records, cache, device, cfg["eval"]["ensemble_weights"]) if len(val_records.eeg) else {"top1_acc": 0.0, "top5_acc": 0.0}
        row = {
            "stage": stage,
            "epoch": epoch,
            "train_loss": loss_sum / max(1, n_seen),
            "val_top1": metrics["top1_acc"],
            "val_top5": metrics["top5_acc"],
            "lr": float(opt.param_groups[0]["lr"]),
            "seconds": time.perf_counter() - start,
            "um_hard": um_acc["hard"] / max(1, n_seen),
            "um_normal": um_acc["normal"] / max(1, n_seen),
            "um_easy": um_acc["easy"] / max(1, n_seen),
        }
        hist.append(row)
        print(f"{stage} epoch {epoch:03d}/{epochs:03d} loss={row['train_loss']:.4f} val@1={row['val_top1']:.4f} val@5={row['val_top5']:.4f}", flush=True)
        if row["val_top1"] >= best:
            best = row["val_top1"]
            torch.save(checkpoint_dict(model, cfg, train_records, row), output_dir / "best.pt")
        torch.save(checkpoint_dict(model, cfg, train_records, row), output_dir / "last.pt")
    return best, hist


def checkpoint_dict(model: CogCapProModel, cfg: dict, train_records, metrics: dict) -> dict:
    return {
        "model_state": model.cpu().state_dict() if False else model.state_dict(),
        "channels": int(train_records.eeg.shape[1]),
        "time_steps": int(train_records.eeg.shape[2]),
        "embed_dim": int(cfg["model"]["embed_dim"]),
        "model_config": cfg["model"],
        "config": cfg,
        "train_image_ids": train_records.image_ids,
        "train_concepts": train_records.concepts,
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/cogcappro_rn50.yaml")
    p.add_argument("--data-dir", default="image-eeg-data")
    p.add_argument("--feature-cache", default="cache/features_rn50.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="runs/seed0")
    p.add_argument("--device", default="auto")
    p.add_argument("--full-train", action="store_true")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--init-ckpt", default=None, help="Optional checkpoint to initialize model weights before training.")
    p.add_argument("--epochs", type=int, default=None, help="Override Stage A epochs.")
    p.add_argument("--align-epochs", type=int, default=None, help="Override Stage B epochs.")
    p.add_argument("--finetune-epochs", type=int, default=None, help="Override Stage C epochs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg["seed"] = args.seed
    cfg["data"]["data_dir"] = args.data_dir
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.align_epochs is not None:
        cfg["train"]["align_epochs"] = args.align_epochs
    if args.finetune_epochs is not None:
        cfg["train"]["finetune_epochs"] = args.finetune_epochs
    set_seed(args.seed)
    out_dir = ensure_dir(args.output_dir)
    device = choose_device(args.device)
    cache = load_feature_cache(args.feature_cache)
    train_all = load_eeg_dataset(args.data_dir, "train", avg_trials=bool(cfg["data"]["avg_trials_train"]), selected_channels=cfg["data"].get("selected_channels"), image_root="auto")
    mean, std = train_eeg_stats(train_all)
    train_all = train_all.normalize(mean, std)
    if args.max_train_samples:
        train_all = subset_records(train_all, torch.arange(min(args.max_train_samples, len(train_all.eeg))))
    if args.full_train:
        train_records = train_all
        val_records = load_eeg_dataset(args.data_dir, "train", avg_trials=True, selected_channels=cfg["data"].get("selected_channels"), image_root="auto").normalize(mean, std)
        val_records = subset_records(val_records, torch.arange(min(200, len(val_records.eeg))))
    else:
        tr_idx, va_idx = deterministic_group_split(train_all, 0.1, args.seed)
        train_records = subset_records(train_all, tr_idx)
        val_records = subset_records(train_all, va_idx)
    model = CogCapProModel(
        channels=int(train_records.eeg.shape[1]),
        embed_dim=int(cfg["model"]["embed_dim"]),
        hidden=int(cfg["model"]["expert_hidden"]),
        dropout=float(cfg["model"]["dropout"]),
        fusion_layers=int(cfg["model"]["fusion_layers"]),
        fusion_heads=int(cfg["model"]["fusion_heads"]),
    ).to(device)
    if args.init_ckpt:
        init = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(init["model_state"], strict=True)
        print(f"Initialized model from {args.init_ckpt}", flush=True)
    history: list[dict] = []
    _, h = run_stage(model, train_records, val_records, cache, cfg, device, out_dir, stage="A", epochs=int(cfg["train"]["epochs"]), lr=float(cfg["train"]["lr"]))
    history.extend(h)
    if int(cfg["train"].get("align_epochs", 0)) > 0:
        _, h = run_stage(model, train_records, val_records, cache, cfg, device, out_dir, stage="B", epochs=int(cfg["train"]["align_epochs"]), lr=float(cfg["train"]["finetune_lr"]), freeze_experts=True)
        history.extend(h)
    if int(cfg["train"].get("finetune_epochs", 0)) > 0:
        for p in model.parameters():
            p.requires_grad = True
        _, h = run_stage(model, train_records, val_records, cache, cfg, device, out_dir, stage="C", epochs=int(cfg["train"]["finetune_epochs"]), lr=float(cfg["train"]["finetune_lr"]))
        history.extend(h)
    write_json(out_dir / "metrics.json", history)
    print(f"Wrote checkpoints and metrics to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
