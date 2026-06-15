from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .features import l2_normalize


def contrastive_loss(brain_embeddings: torch.Tensor, image_embeddings: torch.Tensor, temperature: float) -> torch.Tensor:
    brain_embeddings = l2_normalize(brain_embeddings)
    image_embeddings = l2_normalize(image_embeddings)
    logits = brain_embeddings @ image_embeddings.T / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    loss_i = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.T, targets)
    return 0.5 * (loss_i + loss_t)


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    *,
    topk: int = 10,
) -> torch.Tensor:
    embeddings = l2_normalize(embeddings)
    labels = labels.to(embeddings.device)
    logits = embeddings @ embeddings.T / temperature
    batch_size = logits.shape[0]
    eye = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
    same_label = labels[:, None] == labels[None, :]
    positive = same_label & ~eye
    if topk > 0 and topk < batch_size:
        top_idx = logits.masked_fill(eye, -torch.inf).topk(k=min(topk, batch_size - 1), dim=1).indices
        top_mask = torch.zeros_like(positive)
        top_mask.scatter_(1, top_idx, True)
        positive = positive & top_mask
    valid = positive.any(dim=1)
    if not bool(valid.any()):
        return embeddings.new_tensor(0.0)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits).masked_fill(eye, 0.0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    mean_log_prob = (positive.float() * log_prob).sum(dim=1) / positive.float().sum(dim=1).clamp_min(1.0)
    return -mean_log_prob[valid].mean()


def train_retrieval_model(
    model: torch.nn.Module,
    eeg: torch.Tensor,
    image_features: torch.Tensor,
    labels: torch.Tensor | None = None,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    temperature: float,
    device: torch.device,
    scm_weight: float = 0.0,
    scm_topk: int = 10,
    verbose: bool = False,
) -> list[dict[str, float]]:
    if labels is None:
        labels = torch.arange(eeg.shape[0])
    dataset = TensorDataset(eeg.float(), image_features.float(), labels.long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict[str, float]] = []
    model.to(device)

    for epoch in range(1, epochs + 1):
        start_time = time.perf_counter()
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch_eeg, batch_img, batch_labels in loader:
            batch_eeg = batch_eeg.to(device)
            batch_img = batch_img.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_eeg)
            loss = contrastive_loss(pred, batch_img, temperature)
            if scm_weight > 0:
                loss = loss + scm_weight * supervised_contrastive_loss(
                    pred,
                    batch_labels,
                    temperature,
                    topk=scm_topk,
                )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * batch_eeg.shape[0]
            total_count += batch_eeg.shape[0]
        avg_loss = total_loss / max(total_count, 1)
        elapsed = time.perf_counter() - start_time
        history.append({"epoch": epoch, "train_loss": avg_loss, "seconds": elapsed})
        if verbose:
            print(
                f"epoch {epoch:03d}/{epochs:03d} | "
                f"loss={avg_loss:.6f} | "
                f"samples={total_count} | "
                f"time={elapsed:.1f}s",
                flush=True,
            )
    return history
