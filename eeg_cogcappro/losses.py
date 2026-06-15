from __future__ import annotations

import torch
import torch.nn.functional as F

from .utils import l2_normalize


def symmetric_clip_loss(eeg_emb: torch.Tensor, target_emb: torch.Tensor, logit_scale: torch.Tensor | float) -> torch.Tensor:
    eeg_emb = l2_normalize(eeg_emb)
    target_emb = l2_normalize(target_emb)
    scale = logit_scale.exp() if isinstance(logit_scale, torch.Tensor) and logit_scale.ndim == 0 else logit_scale
    logits = scale * eeg_emb @ target_emb.T
    targets = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))


def _one_way_scm(a: torch.Tensor, b: torch.Tensor, labels: torch.Tensor, top_k: int, tau: float) -> torch.Tensor:
    logits = l2_normalize(a) @ l2_normalize(b).T / tau
    labels = labels.to(logits.device)
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    same = same | eye
    if top_k > 0 and top_k < logits.shape[1]:
        masked = logits.masked_fill(~same, -torch.inf)
        vals, idx = masked.topk(k=min(top_k, logits.shape[1]), dim=1)
        pos = torch.zeros_like(same)
        pos.scatter_(1, idx, torch.isfinite(vals))
        pos = pos | eye
    else:
        pos = same
    denom = torch.logsumexp(logits, dim=1)
    numer = torch.logsumexp(logits.masked_fill(~pos, -torch.inf), dim=1)
    return (denom - numer).mean()


def scm_loss(eeg_emb: torch.Tensor, target_emb: torch.Tensor, labels: torch.Tensor, top_k: int = 10, tau: float = 0.07) -> torch.Tensor:
    return 0.5 * (_one_way_scm(eeg_emb, target_emb, labels, top_k, tau) + _one_way_scm(target_emb, eeg_emb, labels, top_k, tau))


def sth_align_loss(
    pred: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    lambda_mse: float = 1.0,
    lambda_cos: float = 0.5,
    lambda_reg: float = 1e-4,
) -> torch.Tensor:
    total = None
    for key, y in targets.items():
        if key not in pred:
            continue
        x = pred[key]
        loss = lambda_mse * F.mse_loss(x, y)
        loss = loss + lambda_cos * (1.0 - F.cosine_similarity(x, y, dim=-1).mean())
        loss = loss + lambda_reg * x.pow(2).mean()
        total = loss if total is None else total + loss
    if total is None:
        raise ValueError("No overlapping keys for STH alignment loss")
    return total
