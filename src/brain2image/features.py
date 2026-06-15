from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)


def stable_hash_embedding(text: str, dim: int) -> torch.Tensor:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**31)
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(dim, generator=generator)


def _simple_image_vector(path: Path, image_size: int) -> Optional[torch.Tensor]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required for image feature extraction.") from exc

    if not path.exists():
        return None
    with Image.open(path) as img:
        img = img.convert("RGB").resize((image_size, image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).reshape(-1)


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: Sequence[Optional[Path]], image_ids: Sequence[str], transform) -> None:
        self.image_paths = list(image_paths)
        self.image_ids = list(image_ids)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, bool, str]:
        from PIL import Image

        path = self.image_paths[index]
        if path is None or not path.exists():
            return torch.zeros(3, 224, 224), False, self.image_ids[index]
        with Image.open(path) as img:
            return self.transform(img.convert("RGB")), True, self.image_ids[index]


def _torchvision_feature_model(name: str, device: torch.device):
    from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50

    if name in {"torchvision-rn18", "torchvision-rn18-logits"}:
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
    elif name in {"torchvision-rn50", "torchvision-rn50-logits"}:
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported torchvision backend '{name}'.")
    if not name.endswith("-logits"):
        model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model, weights.transforms()


def _build_torchvision_features(
    image_ids: Sequence[str],
    image_paths: Sequence[Optional[Path]],
    *,
    backend: str,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    model, transform = _torchvision_feature_model(backend, device)
    dataset = ImagePathDataset(image_paths, image_ids, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    vectors: list[torch.Tensor] = []
    with torch.no_grad():
        for images, valid, batch_ids in loader:
            images = images.to(device)
            feats = model(images).float().cpu()
            valid = valid.cpu()
            for row, ok, image_id in zip(feats, valid, batch_ids):
                if bool(ok):
                    vectors.append(row)
                else:
                    vectors.append(stable_hash_embedding(str(image_id), feats.shape[1]))
    return l2_normalize(torch.stack(vectors, dim=0))


def build_image_features(
    image_ids: Sequence[str],
    image_paths: Sequence[Optional[Path]],
    *,
    embedding_dim: int,
    backend: str = "simple",
    image_size: int = 32,
    device: torch.device | str = "cpu",
    batch_size: int = 64,
    cache_path: Optional[Path | str] = None,
) -> torch.Tensor:
    if backend not in {
        "simple",
        "hash",
        "torchvision-rn18",
        "torchvision-rn50",
        "torchvision-rn18-logits",
        "torchvision-rn50-logits",
    }:
        raise ValueError(
            f"Unsupported feature backend '{backend}'. Use 'simple', 'hash', torchvision feature, or torchvision logits backends."
        )
    cache = Path(cache_path) if cache_path is not None else None
    if cache is not None and cache.exists():
        loaded = torch.load(cache, map_location="cpu", weights_only=False)
        if (
            loaded.get("backend") == backend
            and loaded.get("image_ids") == list(image_ids)
            and "features" in loaded
        ):
            return l2_normalize(loaded["features"].to(device))

    if backend.startswith("torchvision-"):
        features = _build_torchvision_features(
            image_ids,
            image_paths,
            backend=backend,
            device=torch.device(device),
            batch_size=batch_size,
        )
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"backend": backend, "image_ids": list(image_ids), "features": features.cpu()}, cache)
        return l2_normalize(features.to(device))

    vectors: list[torch.Tensor] = []
    projection: Optional[torch.Tensor] = None
    if backend == "simple":
        input_dim = 3 * image_size * image_size
        generator = torch.Generator().manual_seed(0)
        projection = torch.randn(input_dim, embedding_dim, generator=generator) / np.sqrt(input_dim)

    for image_id, path in zip(image_ids, image_paths):
        vec: Optional[torch.Tensor] = None
        if backend == "simple" and path is not None and projection is not None:
            raw = _simple_image_vector(path, image_size)
            if raw is not None:
                vec = raw @ projection
        if vec is None:
            vec = stable_hash_embedding(image_id, embedding_dim)
        vectors.append(vec.float())

    features = l2_normalize(torch.stack(vectors, dim=0))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"backend": backend, "image_ids": list(image_ids), "features": features.cpu()}, cache)
    return features.to(device)
