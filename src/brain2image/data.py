from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch


@dataclass(frozen=True)
class EEGSplit:
    eeg: torch.Tensor
    labels: list[int]
    texts: list[str]
    image_ids: list[str]
    raw_images: list[str]
    image_paths: list[Optional[Path]]


def selected_channel_indices_from_jsonl(
    selected_channels: Union[str, Sequence[str]],
    eeg_channel_jsonl: Union[str, Path],
) -> list[int]:
    if isinstance(selected_channels, str):
        selected_channels = [selected_channels]
    selected_channels = list(selected_channels)

    channel_names: list[str] = []
    with Path(eeg_channel_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            name = item.get("name") or item.get("channel_name") or item.get("label")
            if name is None:
                raise KeyError(
                    "Each channel JSONL row must contain one of: name, channel_name, label."
                )
            channel_names.append(str(name))

    name_to_index = {name: idx for idx, name in enumerate(channel_names)}
    missing = [ch for ch in selected_channels if ch not in name_to_index]
    if missing:
        raise ValueError(f"Unknown EEG channels: {missing}")
    return [name_to_index[ch] for ch in selected_channels]


def _normalize_raw_images(imgs: np.ndarray, avg_trials: bool, n_eeg: int) -> list[str]:
    if avg_trials:
        if imgs.ndim == 2:
            imgs = imgs[:, 0]
        imgs = imgs.reshape(-1)[:n_eeg]
    else:
        imgs = imgs.reshape(-1)

    out: list[str] = []
    for value in imgs.tolist():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        out.append(str(value))
    if len(out) != n_eeg:
        raise ValueError(f"EEG/image mismatch: {n_eeg} EEG samples vs {len(out)} image ids.")
    return out


def _normalize_repeated_array(values: np.ndarray, avg_trials: bool, n_eeg: int) -> list:
    if avg_trials:
        if values.ndim == 2:
            values = values[:, 0]
        values = values.reshape(-1)[:n_eeg]
    else:
        values = values.reshape(-1)
    out = values.tolist()
    if len(out) != n_eeg:
        raise ValueError(f"EEG/metadata mismatch: {n_eeg} EEG samples vs {len(out)} values.")
    return out


def _candidate_image_roots(data_dir: Path, extra_roots: Optional[Sequence[Union[str, Path]]]) -> list[Path]:
    roots: list[Path] = []
    if extra_roots:
        roots.extend(Path(root) for root in extra_roots)
    roots.extend(
        [
            data_dir,
            data_dir / "images",
            data_dir / "image",
            data_dir / "stimuli",
            data_dir / "stimulus_images",
            data_dir / "things_images",
        ]
    )
    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        root = root.expanduser()
        if root not in seen and root.exists():
            unique.append(root)
            seen.add(root)
    return unique


def _index_images(roots: Sequence[Path]) -> dict[str, Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    index: dict[str, Path] = {}
    for root in roots:
        if root.is_file() and root.suffix.lower() in suffixes:
            index.setdefault(root.stem, root)
            index.setdefault(root.name, root)
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in suffixes:
                index.setdefault(path.stem, path)
                index.setdefault(path.name, path)
    return index


def resolve_image_paths(
    raw_images: Sequence[str],
    data_dir: Union[str, Path],
    image_roots: Optional[Sequence[Union[str, Path]]] = None,
) -> list[Optional[Path]]:
    data_dir = Path(data_dir)
    roots = _candidate_image_roots(data_dir, image_roots)
    image_index = _index_images(roots)

    resolved: list[Optional[Path]] = []
    for raw in raw_images:
        raw_path = Path(raw)
        candidates = [
            raw_path,
            data_dir / raw_path,
            data_dir / raw_path.name,
        ]
        found = next((p for p in candidates if p.exists() and p.is_file()), None)
        if found is None:
            found = image_index.get(raw_path.name) or image_index.get(raw_path.stem)
        resolved.append(found)
    return resolved


def load_eeg_split(
    data_dir: Union[str, Path],
    split: str,
    *,
    avg_trials: bool = True,
    selected_channels: Optional[Union[str, Sequence[str]]] = None,
    eeg_channel_jsonl: Optional[Union[str, Path]] = None,
    image_roots: Optional[Sequence[Union[str, Path]]] = None,
) -> EEGSplit:
    data_dir = Path(data_dir)
    pt_path = data_dir / f"{split}.pt"
    if not pt_path.exists():
        raise FileNotFoundError(
            f"Missing {pt_path}. Expected a Project1 data directory with {split}.pt."
        )

    loaded = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    if "eeg" not in loaded:
        raise KeyError(f"{pt_path} must contain key 'eeg'. Found keys: {sorted(loaded.keys())}")
    if "img" not in loaded:
        raise KeyError(f"{pt_path} must contain key 'img'. Found keys: {sorted(loaded.keys())}")

    x = torch.as_tensor(loaded["eeg"]).float()
    if x.ndim == 4:
        x = x.mean(dim=1) if avg_trials else x.reshape(-1, *x.shape[2:])
    elif x.ndim != 3:
        raise ValueError(f"Unexpected EEG shape in {pt_path}: {tuple(x.shape)}")

    if selected_channels is not None:
        channel_file = Path(eeg_channel_jsonl) if eeg_channel_jsonl else data_dir / "EEG_CHANNELS.jsonl"
        indices = selected_channel_indices_from_jsonl(selected_channels, channel_file)
        x = x[:, indices, :]

    raw_images = _normalize_raw_images(np.array(loaded["img"]), avg_trials, x.shape[0])
    raw_labels = _normalize_repeated_array(np.array(loaded.get("label", np.arange(x.shape[0]))), avg_trials, x.shape[0])
    raw_texts = _normalize_repeated_array(np.array(loaded.get("text", raw_images)), avg_trials, x.shape[0])
    labels = [int(item) for item in raw_labels]
    texts = [str(item.decode("utf-8") if isinstance(item, bytes) else item) for item in raw_texts]
    image_ids = [Path(item).stem for item in raw_images]
    image_paths = resolve_image_paths(raw_images, data_dir, image_roots)
    return EEGSplit(
        eeg=x.contiguous(),
        labels=labels,
        texts=texts,
        image_ids=image_ids,
        raw_images=raw_images,
        image_paths=image_paths,
    )
