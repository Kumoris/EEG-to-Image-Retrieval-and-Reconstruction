from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .image_resolver import ImageResolver
from .utils import safe_torch_load

CONCEPT_RE = re.compile(r"_\d+[A-Za-z]*$")


def concept_from_image_id(image_id: str) -> str:
    return CONCEPT_RE.sub("", Path(str(image_id)).stem)


def selected_channel_indices(selected_channels: Sequence[str], channel_jsonl: str | Path) -> list[int]:
    names: list[str] = []
    with Path(channel_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            names.append(str(item.get("name") or item.get("channel_name") or item.get("label")))
    lookup = {name: i for i, name in enumerate(names)}
    missing = [ch for ch in selected_channels if ch not in lookup]
    if missing:
        raise ValueError(f"Unknown EEG channels: {missing}")
    return [lookup[ch] for ch in selected_channels]


@dataclass
class EEGRecords:
    eeg: torch.Tensor
    image_ids: list[str]
    concepts: list[str]
    labels: torch.Tensor
    texts: list[str]
    raw_images: list[str]
    image_paths: list[Path | None]
    indices: torch.Tensor

    def normalize(self, mean: torch.Tensor, std: torch.Tensor) -> "EEGRecords":
        return EEGRecords(
            eeg=((self.eeg - mean) / std.clamp_min(1e-6)).contiguous(),
            image_ids=self.image_ids,
            concepts=self.concepts,
            labels=self.labels,
            texts=self.texts,
            raw_images=self.raw_images,
            image_paths=self.image_paths,
            indices=self.indices,
        )


class EEGDataset(Dataset):
    def __init__(self, records: EEGRecords, transform=None) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return self.records.eeg.shape[0]

    def __getitem__(self, idx: int) -> dict:
        eeg = self.records.eeg[idx]
        if self.transform is not None:
            eeg = self.transform(eeg)
        return {
            "eeg": eeg,
            "image_id": self.records.image_ids[idx],
            "concept": self.records.concepts[idx],
            "label": self.records.labels[idx],
            "index": self.records.indices[idx],
        }


def _array_to_list(values: np.ndarray, avg_trials: bool, n: int) -> list:
    if avg_trials:
        if values.ndim == 2:
            values = values[:, 0]
        values = values.reshape(-1)[:n]
    else:
        values = values.reshape(-1)
    out = values.tolist()
    if len(out) != n:
        raise ValueError(f"metadata length mismatch: {len(out)} vs {n}")
    return [x.decode("utf-8") if isinstance(x, bytes) else x for x in out]


def build_concept_labels(concepts: Sequence[str], existing: dict[str, int] | None = None) -> tuple[torch.Tensor, dict[str, int]]:
    mapping = dict(existing or {})
    for concept in concepts:
        if concept not in mapping:
            mapping[concept] = len(mapping)
    return torch.tensor([mapping[c] for c in concepts], dtype=torch.long), mapping


def load_eeg_dataset(
    data_directory: str | Path,
    split: str,
    *,
    avg_trials: bool = True,
    selected_channels: Sequence[str] | None = None,
    concept_to_label: dict[str, int] | None = None,
    image_root: str | Path | Sequence[str | Path] | None = "auto",
) -> EEGRecords:
    data_dir = Path(data_directory)
    loaded = safe_torch_load(data_dir / f"{split}.pt", map_location="cpu")
    x = torch.as_tensor(loaded["eeg"]).float()
    if x.ndim == 4:
        x = x.mean(dim=1) if avg_trials else x.reshape(-1, *x.shape[2:])
    elif x.ndim != 3:
        raise ValueError(f"Unexpected EEG shape: {tuple(x.shape)}")
    if selected_channels:
        channel_file = data_dir / "THINGS_EEG_CHANNELS.jsonl"
        if not channel_file.exists():
            channel_file = data_dir / "EEG_CHANNELS.jsonl"
        x = x[:, selected_channel_indices(list(selected_channels), channel_file), :]
    raw_images = [str(v) for v in _array_to_list(np.asarray(loaded["img"]), avg_trials, x.shape[0])]
    texts = [str(v) for v in _array_to_list(np.asarray(loaded.get("text", loaded["img"])), avg_trials, x.shape[0])]
    image_ids = [Path(v).stem for v in raw_images]
    concepts = [concept_from_image_id(v) for v in image_ids]
    labels, _ = build_concept_labels(concepts, concept_to_label)
    resolver = ImageResolver(data_dir, image_root=image_root)
    paths = resolver.resolve_many(raw_images, warn=False)
    missing = [image_ids[i] for i, p in enumerate(paths) if p is None]
    if missing:
        print(f"Warning: {split} has {len(missing)} unresolved images; feature code will use text/hash fallback.", flush=True)
    return EEGRecords(
        eeg=x.contiguous(),
        image_ids=image_ids,
        concepts=concepts,
        labels=labels,
        texts=texts,
        raw_images=raw_images,
        image_paths=paths,
        indices=torch.arange(x.shape[0], dtype=torch.long),
    )


def train_eeg_stats(records: EEGRecords) -> tuple[torch.Tensor, torch.Tensor]:
    mean = records.eeg.mean(dim=(0, 2), keepdim=True)
    std = records.eeg.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
    return mean, std


def deterministic_group_split(records: EEGRecords, val_fraction: float = 0.1, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    concepts = sorted(set(records.concepts))
    perm = torch.randperm(len(concepts), generator=gen).tolist()
    n_val = max(1, int(round(len(concepts) * val_fraction)))
    val_concepts = {concepts[i] for i in perm[:n_val]}
    train_idx = [i for i, c in enumerate(records.concepts) if c not in val_concepts]
    val_idx = [i for i, c in enumerate(records.concepts) if c in val_concepts]
    return torch.tensor(train_idx, dtype=torch.long), torch.tensor(val_idx, dtype=torch.long)


def subset_records(records: EEGRecords, indices: torch.Tensor) -> EEGRecords:
    idx = indices.tolist()
    return EEGRecords(
        eeg=records.eeg[indices],
        image_ids=[records.image_ids[i] for i in idx],
        concepts=[records.concepts[i] for i in idx],
        labels=records.labels[indices],
        texts=[records.texts[i] for i in idx],
        raw_images=[records.raw_images[i] for i in idx],
        image_paths=[records.image_paths[i] for i in idx],
        indices=records.indices[indices],
    )
