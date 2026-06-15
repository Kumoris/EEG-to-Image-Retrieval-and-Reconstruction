from __future__ import annotations

import torch


def compute_retrieval_metrics(logits: torch.Tensor, targets: torch.Tensor | None = None) -> dict[str, float]:
    if logits.ndim != 2:
        raise ValueError(f"Expected [num_queries, num_candidates] logits, got {tuple(logits.shape)}")
    if targets is None:
        if logits.shape[0] != logits.shape[1]:
            raise ValueError("targets must be provided when logits are not square.")
        targets = torch.arange(logits.shape[0], device=logits.device)
    targets = targets.to(logits.device)
    top1 = logits.argmax(dim=1)
    k = min(5, logits.shape[1])
    topk = logits.topk(k=k, dim=1).indices
    return {
        "top1_acc": (top1 == targets).float().mean().item(),
        "top5_acc": (topk == targets[:, None]).any(dim=1).float().mean().item(),
    }


def ranks_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.argsort(logits, dim=1, descending=True)

