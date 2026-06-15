from __future__ import annotations

import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1, eps=1e-8)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to read config files. Install pyyaml.") from exc
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def compute_retrieval_metrics(logits: torch.Tensor, targets: torch.Tensor | None = None) -> dict[str, float]:
    if logits.ndim != 2:
        raise ValueError(f"Expected [N, M] logits, got {tuple(logits.shape)}")
    if targets is None:
        if logits.shape[0] != logits.shape[1]:
            raise ValueError("targets must be provided for non-square logits")
        targets = torch.arange(logits.shape[0], device=logits.device)
    targets = targets.to(logits.device)
    top1 = logits.argmax(dim=1)
    k = min(5, logits.shape[1])
    topk = logits.topk(k=k, dim=1).indices
    return {
        "top1_acc": float((top1 == targets).float().mean().item()),
        "top5_acc": float((topk == targets[:, None]).any(dim=1).float().mean().item()),
    }


def summarize_metric_dicts(items: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = sorted({k for item in items for k, v in item.items() if isinstance(v, (int, float))})
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.array([float(item[key]) for item in items if key in item], dtype=np.float64)
        out[key] = {
            "mean": float(values.mean()) if values.size else math.nan,
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        }
    return out


def safe_torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    return torch.load(str(path), map_location=map_location, weights_only=False)


def autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")
