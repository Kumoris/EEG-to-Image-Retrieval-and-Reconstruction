from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .data import build_concept_labels, load_eeg_dataset
from .image_resolver import load_rgb
from .utils import ensure_dir, l2_normalize, safe_torch_load


class _RandomAugmentor:
    def __init__(self, crop_scale=(0.5, 1.0), color_jitter=(0.3, 0.3, 0.3, 0.1), hflip=True):
        self.crop_scale = crop_scale
        self.color_jitter = color_jitter
        self.hflip = hflip

    def __call__(self, img: Image.Image, size: int = 224) -> Image.Image:
        w, h = img.size
        scale = np.random.uniform(*self.crop_scale)
        crop_w, crop_h = int(w * scale), int(h * scale)
        x0 = np.random.randint(0, max(1, w - crop_w + 1))
        y0 = np.random.randint(0, max(1, h - crop_h + 1))
        img = img.crop((x0, y0, x0 + crop_w, y0 + crop_h)).resize((size, size), Image.BICUBIC)
        if self.color_jitter:
            bri, con, sat, hue = self.color_jitter
            if np.random.rand() < 0.8:
                img = ImageEnhance.Brightness(img).enhance(1.0 + np.random.uniform(-bri, bri))
            if np.random.rand() < 0.8:
                img = ImageEnhance.Contrast(img).enhance(1.0 + np.random.uniform(-con, con))
            if np.random.rand() < 0.8:
                img = ImageEnhance.Color(img).enhance(1.0 + np.random.uniform(-sat, sat))
        if self.hflip and np.random.rand() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


def stable_hash_embedding(text: str, dim: int = 512) -> torch.Tensor:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little") % (2**31)
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(dim, generator=gen)


def foveated_blur(img: Image.Image, sigma: float) -> Image.Image:
    if sigma <= 0:
        return img.copy()
    img = img.convert("RGB")
    blur = img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))
    w, h = img.size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt(((xx - w / 2) / max(w, 1)) ** 2 + ((yy - h / 2) / max(h, 1)) ** 2)
    mask = np.clip((dist / 0.55) ** 1.7, 0, 1)
    a = np.asarray(img, dtype=np.float32)
    b = np.asarray(blur, dtype=np.float32)
    out = a * (1 - mask[..., None]) + b * mask[..., None]
    return Image.fromarray(np.uint8(np.clip(out, 0, 255)))


def edge_image(img: Image.Image) -> Image.Image:
    gray = np.asarray(ImageOps.grayscale(img).filter(ImageFilter.GaussianBlur(radius=1.0)), dtype=np.float32) / 255.0
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    mag = np.sqrt(gx * gx + gy * gy)
    mag = (mag > np.quantile(mag, 0.75)).astype(np.float32) * 255
    return Image.fromarray(mag.astype(np.uint8)).convert("RGB")


def depth_proxy_image(img: Image.Image) -> Image.Image:
    gray = np.asarray(ImageOps.grayscale(img), dtype=np.float32) / 255.0
    gy = np.zeros_like(gray)
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    proxy = gray * 0.7 + (1.0 - np.linspace(0, 1, gray.shape[0], dtype=np.float32)[:, None]) * 0.2 + np.abs(gy) * 0.1
    proxy = (proxy - proxy.min()) / max(float(proxy.max() - proxy.min()), 1e-6)
    return Image.fromarray(np.uint8(proxy * 255)).convert("RGB")


class ClipBackend:
    def __init__(self, backbone: str, device: torch.device, pretrained: str = "openai") -> None:
        self.device = device
        self.name = backbone
        self.kind = "hash"
        self.model = None
        self.preprocess: Optional[Callable] = None
        self.tokenizer = None
        self._projections: dict[tuple[int, int], torch.Tensor] = {}
        try:
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=pretrained)
            self.model = model.eval().to(device)
            self.preprocess = preprocess
            self.tokenizer = open_clip.get_tokenizer(backbone)
            self.kind = "open_clip"
            return
        except Exception as exc:
            print(f"Warning: open_clip unavailable for {backbone}: {exc}", flush=True)
        try:
            import clip

            model, preprocess = clip.load(backbone, device=device)
            self.model = model.eval()
            self.preprocess = preprocess
            self.tokenizer = clip.tokenize
            self.kind = "clip"
        except Exception as exc:
            print(f"Warning: openai clip unavailable for {backbone}: {exc}", flush=True)
        try:
            from torchvision.models import ResNet50_Weights, resnet50

            weights = ResNet50_Weights.DEFAULT
            model = resnet50(weights=weights)
            model.fc = torch.nn.Identity()
            self.model = model.eval().to(device)
            self.preprocess = weights.transforms()
            self.kind = "torchvision-rn50"
            print("Warning: using torchvision RN50 visual fallback; text features use deterministic prompt hashes.", flush=True)
        except Exception as exc:
            print(f"Warning: torchvision RN50 fallback unavailable: {exc}", flush=True)
            print("Warning: using deterministic hash fallback features. Install open_clip_torch with weights for best results.", flush=True)

    @torch.no_grad()
    def _fit_dim(self, feats: torch.Tensor, dim: int) -> torch.Tensor:
        feats = feats.float().cpu()
        if feats.shape[1] == dim:
            return l2_normalize(feats)
        key = (int(feats.shape[1]), int(dim))
        if key not in self._projections:
            gen = torch.Generator().manual_seed(17_013 + key[0] * 13 + key[1])
            self._projections[key] = torch.randn(key[0], key[1], generator=gen) / np.sqrt(key[0])
        return l2_normalize(feats @ self._projections[key])

    @torch.no_grad()
    def encode_images(self, images: Sequence[Image.Image], ids: Sequence[str], dim: int = 512) -> torch.Tensor:
        if self.kind == "hash" or self.model is None or self.preprocess is None:
            return l2_normalize(torch.stack([stable_hash_embedding(f"image:{i}", dim) for i in ids]))
        batch = torch.stack([self.preprocess(img).to(self.device) for img in images])
        feats = self.model(batch).float() if self.kind == "torchvision-rn50" else self.model.encode_image(batch).float()
        return self._fit_dim(feats, dim)

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str], dim: int = 512) -> torch.Tensor:
        if self.kind == "hash" or self.model is None or self.tokenizer is None:
            return l2_normalize(torch.stack([stable_hash_embedding(f"text:{t}", dim) for t in texts]))
        toks = self.tokenizer(list(texts)).to(self.device)
        feats = self.model.encode_text(toks).float()
        return self._fit_dim(feats, dim)


class DINOv2Backend:
    def __init__(self, model_name: str = "dinov2_vitb14", device: torch.device = torch.device("cpu"), num_aug: int = 2) -> None:
        self.device = device
        self.model_name = model_name
        self.num_aug = num_aug
        self.kind = "hash"
        self.model = None
        self.preprocess: Optional[Callable] = None
        self._projections: dict[tuple[int, int], torch.Tensor] = {}
        try:
            dinov2 = torch.hub.load("facebookresearch/dinov2", model_name)
            dinov2 = dinov2.eval().to(device)
            self.model = dinov2
            from torchvision import transforms as T
            self.preprocess = T.Compose([
                T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.kind = "dinov2"
            print(f"Loaded DINOv2 model: {model_name}", flush=True)
        except Exception as exc:
            print(f"Warning: DINOv2 ({model_name}) unavailable via torch.hub: {exc}", flush=True)
            try:
                from transformers import AutoModel, AutoImageProcessor
                hf_id = {"dinov2_vits14": "facebook/dinov2-small", "dinov2_vitb14": "facebook/dinov2-base",
                         "dinov2_vitl14": "facebook/dinov2-large", "dinov2_vitg14": "facebook/dinov2-giant"}.get(model_name, "facebook/dinov2-base")
                processor = AutoImageProcessor.from_pretrained(hf_id)
                model = AutoModel.from_pretrained(hf_id)
                model = model.eval().to(device)
                self.model = model
                self.preprocess = processor
                self.kind = "dinov2_hf"
                print(f"Loaded DINOv2 via transformers: {hf_id}", flush=True)
            except Exception as exc2:
                print(f"Warning: DINOv2 also unavailable via transformers: {exc2}", flush=True)
                print("Warning: using hash fallback for DINOv2 features.", flush=True)
        self._augmentor = _RandomAugmentor() if self.num_aug > 1 else None

    @torch.no_grad()
    def _fit_dim(self, feats: torch.Tensor, dim: int) -> torch.Tensor:
        feats = feats.float().cpu()
        if feats.shape[1] == dim:
            return l2_normalize(feats)
        key = (int(feats.shape[1]), int(dim))
        if key not in self._projections:
            gen = torch.Generator().manual_seed(42_017 + key[0] * 7 + key[1])
            self._projections[key] = torch.randn(key[0], key[1], generator=gen) / np.sqrt(key[0])
        return l2_normalize(feats @ self._projections[key])

    @torch.no_grad()
    def _encode_single(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.kind == "dinov2":
            return self.model(pixel_values).float()
        elif self.kind == "dinov2_hf":
            out = self.model(pixel_values)
            if hasattr(out, "last_hidden_state"):
                cls_token = out.last_hidden_state[:, 0]
                return cls_token.float()
            return out.pooler_output.float() if hasattr(out, "pooler_output") else out.last_hidden_state[:, 0].float()
        return None

    @torch.no_grad()
    def encode_images(self, images: Sequence[Image.Image], ids: Sequence[str], dim: int = 768) -> torch.Tensor:
        if self.kind == "hash" or self.model is None:
            return l2_normalize(torch.stack([stable_hash_embedding(f"dinov2:{i}", dim) for i in ids]))
        if self.num_aug <= 1:
            batch = torch.stack([self.preprocess(img).to(self.device) if self.kind == "dinov2" else self.preprocess(img, return_tensors="pt")["pixel_values"].squeeze(0).to(self.device) for img in images])
            if self.kind == "dinov2_hf":
                batch = torch.stack([self.preprocess(img, return_tensors="pt")["pixel_values"].squeeze(0).to(self.device) for img in images])
            feats = self._encode_single(batch)
            return self._fit_dim(feats, dim)

        all_feats = []
        num_passes = max(2, self.num_aug)
        for aug_idx in range(num_passes):
            aug_images = []
            for img in images:
                if aug_idx == 0:
                    aug_images.append(img)
                else:
                    aug_images.append(self._augmentor(img, 224))
            if self.kind == "dinov2":
                batch = torch.stack([self.preprocess(img) for img in aug_images]).to(self.device)
            else:
                batch = torch.stack([self.preprocess(img, return_tensors="pt")["pixel_values"].squeeze(0) for img in aug_images]).to(self.device)
            feats = self._encode_single(batch)
            all_feats.append(feats.cpu())
        avg_feats = torch.stack(all_feats, dim=0).mean(dim=0)
        return self._fit_dim(avg_feats, dim)

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str], dim: int = 768) -> torch.Tensor:
        return l2_normalize(torch.stack([stable_hash_embedding(f"dinov2_text:{t}", dim) for t in texts]))


class VAEBackend:
    def __init__(self, vae_name: str = "stabilityai/sd-vae-ft-mse", device: torch.device = torch.device("cpu")) -> None:
        self.device = device
        self.vae_name = vae_name
        self.kind = "hash"
        self.model = None
        self.preprocess: Optional[Callable] = None
        self._projections: dict[tuple[int, int], torch.Tensor] = {}
        self.latent_channels = 4
        self.latent_spatial = 64
        try:
            from diffusers import AutoencoderKL
            from torchvision import transforms as T
            vae = AutoencoderKL.from_pretrained(vae_name, torch_dtype=torch.float32)
            vae = vae.eval().to(device)
            self.model = vae
            self.preprocess = T.Compose([
                T.Resize(512, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(512),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            self.kind = "sd_vae"
            print(f"Loaded VAE: {vae_name}", flush=True)
        except Exception as exc:
            print(f"Warning: VAE ({vae_name}) unavailable: {exc}", flush=True)
            print("Warning: using hash fallback for VAE latent features.", flush=True)

    @torch.no_grad()
    def _fit_dim(self, feats: torch.Tensor, dim: int) -> torch.Tensor:
        feats = feats.float().cpu()
        if feats.shape[1] == dim:
            return l2_normalize(feats)
        key = (int(feats.shape[1]), int(dim))
        if key not in self._projections:
            gen = torch.Generator().manual_seed(99_007 + key[0] * 17 + key[1])
            self._projections[key] = torch.randn(key[0], key[1], generator=gen) / np.sqrt(key[0])
        return l2_normalize(feats @ self._projections[key])

    @torch.no_grad()
    def encode_images(self, images: Sequence[Image.Image], ids: Sequence[str], dim: int = 512) -> torch.Tensor:
        if self.kind == "hash" or self.model is None:
            return l2_normalize(torch.stack([stable_hash_embedding(f"vae:{i}", dim) for i in ids]))
        batch = torch.stack([self.preprocess(img).to(self.device) for img in images])
        latent_dist = self.model.encode(batch).latent_dist
        latent_mean = latent_dist.mean
        latent_flat = latent_mean.flatten(1)
        return self._fit_dim(latent_flat.cpu(), dim)

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str], dim: int = 512) -> torch.Tensor:
        return l2_normalize(torch.stack([stable_hash_embedding(f"vae_text:{t}", dim) for t in texts]))


def _load_or_placeholder(path: Optional[Path], image_id: str, size: int = 224) -> Image.Image:
    if path is not None and path.exists():
        return load_rgb(path)
    digest = hashlib.sha256(image_id.encode("utf-8")).digest()
    color = tuple(int(v) for v in digest[:3])
    return Image.new("RGB", (size, size), color=color)


def _encode_variant(
    backend: ClipBackend,
    image_ids: list[str],
    paths: list[Optional[Path]],
    variant: str,
    batch_size: int,
    dim: int,
    sigma0: float,
    blur_c: float,
) -> torch.Tensor:
    out: list[torch.Tensor] = []
    for start in range(0, len(image_ids), batch_size):
        ids = image_ids[start : start + batch_size]
        ps = paths[start : start + batch_size]
        imgs = []
        for image_id, path in zip(ids, ps):
            img = _load_or_placeholder(path, image_id)
            if variant == "clean":
                pass
            elif variant == "fovea_low":
                img = foveated_blur(img, max(0.0, sigma0 - blur_c))
            elif variant == "fovea_mid":
                img = foveated_blur(img, sigma0)
            elif variant == "fovea_high":
                img = foveated_blur(img, sigma0 + blur_c)
            elif variant == "edge":
                img = edge_image(img)
            elif variant == "depth":
                img = depth_proxy_image(img)
            imgs.append(img)
        out.append(backend.encode_images(imgs, ids, dim=dim))
    return l2_normalize(torch.cat(out, dim=0))


def _encode_text_batched(backend: ClipBackend, texts: Sequence[str], batch_size: int, dim: int) -> torch.Tensor:
    out: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        out.append(backend.encode_texts(texts[start : start + batch_size], dim=dim))
    return l2_normalize(torch.cat(out, dim=0))


def build_feature_cache(
    data_dir: str | Path,
    image_root: str | Path | None,
    clip_backbone: str,
    output_cache: str | Path,
    *,
    clip_pretrained: str = "openai",
    batch_size: int = 64,
    feature_dim: int = 512,
    sigma0: float = 8.0,
    blur_c: float = 6.0,
    device: str = "auto",
    clean_only: bool = False,
) -> dict:
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    train = load_eeg_dataset(data_dir, "train", avg_trials=True, image_root=image_root)
    test = load_eeg_dataset(data_dir, "test", avg_trials=True, concept_to_label={c: int(l) for c, l in zip(train.concepts, train.labels.tolist())}, image_root=image_root)
    labels, mapping = build_concept_labels(train.concepts + test.concepts)
    n_train = len(train.image_ids)
    image_ids = train.image_ids + test.image_ids
    concepts = train.concepts + test.concepts
    paths = train.image_paths + test.image_paths
    backend = ClipBackend(clip_backbone, dev, pretrained=clip_pretrained)
    texts = [f"a high quality photo of a {c.replace('_', ' ')}" for c in concepts]
    clean = _encode_variant(backend, image_ids, paths, "clean", batch_size, feature_dim, sigma0, blur_c)
    if clean_only:
        features = {
            "image_clean_feature": clean,
            "image_fovea_low": clean,
            "image_fovea_mid": clean,
            "image_fovea_high": clean,
            "edge_feature": clean,
            "depth_feature": clean,
            "text_feature": clean,
        }
    else:
        features = {
            "image_clean_feature": clean,
            "image_fovea_low": _encode_variant(backend, image_ids, paths, "fovea_low", batch_size, feature_dim, sigma0, blur_c),
            "image_fovea_mid": _encode_variant(backend, image_ids, paths, "fovea_mid", batch_size, feature_dim, sigma0, blur_c),
            "image_fovea_high": _encode_variant(backend, image_ids, paths, "fovea_high", batch_size, feature_dim, sigma0, blur_c),
            "edge_feature": _encode_variant(backend, image_ids, paths, "edge", batch_size, feature_dim, sigma0, blur_c),
            "depth_feature": _encode_variant(backend, image_ids, paths, "depth", batch_size, feature_dim, sigma0, blur_c),
            "text_feature": _encode_text_batched(backend, texts, batch_size, feature_dim),
        }
    cache = {
        "image_ids": image_ids,
        "concepts": concepts,
        "labels": labels.long(),
        "split_ranges": {"train": [0, n_train], "test": [n_train, len(image_ids)]},
        "backbone": f"{backend.kind}:{clip_backbone}",
        "preprocessing": {"sigma0": sigma0, "blur_c": blur_c},
        "clean_only": clean_only,
        **features,
    }
    output_cache = Path(output_cache)
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output_cache)
    print(f"Wrote feature cache: {output_cache}", flush=True)
    return cache


def build_multi_feature_cache(
    data_dir: str | Path,
    image_root: str | Path | None,
    output_cache: str | Path,
    *,
    backends: list[str] | None = None,
    dinov2_model: str = "dinov2_vitb14",
    vae_name: str = "stabilityai/sd-vae-ft-mse",
    batch_size: int = 64,
    feature_dim: int = 512,
    device: str = "auto",
) -> dict:
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    if backends is None:
        backbones = ["RN50", "ViT-B-32", "dinov2_da2", "sd_vae"]
    else:
        backbones = list(backends)
    train = load_eeg_dataset(data_dir, "train", avg_trials=True, image_root=image_root)
    test = load_eeg_dataset(data_dir, "test", avg_trials=True, concept_to_label={c: int(l) for c, l in zip(train.concepts, train.labels.tolist())}, image_root=image_root)
    labels, _ = build_concept_labels(train.concepts + test.concepts)
    n_train = len(train.image_ids)
    image_ids = train.image_ids + test.image_ids
    concepts = train.concepts + test.concepts
    paths = train.image_paths + test.image_paths
    texts = [f"a high quality photo of a {c.replace('_', ' ')}" for c in concepts]
    features: dict[str, torch.Tensor] = {}
    backend_info: list[str] = []
    for backbone in backbones:
        print(f"Extracting features for backbone: {backbone}", flush=True)
        if backbone == "dinov2_da2" or backbone.startswith("dinov2"):
            num_aug = 2 if backbone == "dinov2_da2" else 1
            model_name = "dinov2_vitb14" if backbone in ("dinov2_da2", "dinov2") else backbone.replace("dinov2_", "dinov2_")
            backend_obj = DINOv2Backend(model_name=model_name, device=dev, num_aug=num_aug)
            prefix = "dinov2_da2" if backbone == "dinov2_da2" else backbone
            dim = feature_dim
            feats = []
            for start in range(0, len(image_ids), batch_size):
                ids = image_ids[start:start + batch_size]
                ps = paths[start:start + batch_size]
                imgs = [_load_or_placeholder(p, iid) for iid, p in zip(ids, ps)]
                feats.append(backend_obj.encode_images(imgs, ids, dim=dim))
            features[f"{prefix}_feature"] = l2_normalize(torch.cat(feats, dim=0))
            backend_info.append(f"{backend_obj.kind}:{backbone}")
            del backend_obj
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        elif backbone == "sd_vae" or backbone.startswith("sd_vae") or "vae" in backbone:
            backend_obj = VAEBackend(vae_name=vae_name, device=dev)
            prefix = "vae"
            dim = feature_dim
            feats = []
            for start in range(0, len(image_ids), batch_size):
                ids = image_ids[start:start + batch_size]
                ps = paths[start:start + batch_size]
                imgs = [_load_or_placeholder(p, iid) for iid, p in zip(ids, ps)]
                feats.append(backend_obj.encode_images(imgs, ids, dim=dim))
            features[f"{prefix}_feature"] = l2_normalize(torch.cat(feats, dim=0))
            backend_info.append(f"{backend_obj.kind}:{backbone}")
            del backend_obj
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        else:
            clip_pretrained = "openai"
            if backbone == "ViT-L-14" or backbone.startswith("ViT-L"):
                clip_pretrained = "laion2b_s32b_b82k"
            backend_obj = ClipBackend(backbone, dev, pretrained=clip_pretrained)
            safe_name = backbone.lower().replace("/", "_").replace("-", "_")
            dim = feature_dim
            feats = []
            for start in range(0, len(image_ids), batch_size):
                ids = image_ids[start:start + batch_size]
                ps = paths[start:start + batch_size]
                imgs = [_load_or_placeholder(p, iid) for iid, p in zip(ids, ps)]
                feats.append(backend_obj.encode_images(imgs, ids, dim=dim))
            features[f"{safe_name}_feature"] = l2_normalize(torch.cat(feats, dim=0))
            text_feats = _encode_text_batched(backend_obj, texts, batch_size, dim)
            features[f"{safe_name}_text_feature"] = text_feats
            backend_info.append(f"{backend_obj.kind}:{backbone}")
            del backend_obj
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    cache = {
        "image_ids": image_ids,
        "concepts": concepts,
        "labels": labels.long(),
        "split_ranges": {"train": [0, n_train], "test": [n_train, len(image_ids)]},
        "backbones": backend_info,
        "feature_dim": feature_dim,
        **features,
    }
    output_cache = Path(output_cache)
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output_cache)
    print(f"Wrote multi-feature cache: {output_cache}", flush=True)
    print(f"  Backends: {backend_info}", flush=True)
    print(f"  Feature keys: {list(features.keys())}", flush=True)
    return cache


def load_feature_cache(path: str | Path) -> dict:
    cache = safe_torch_load(path, map_location="cpu")
    for key, value in list(cache.items()):
        if key.endswith("_feature") or key.startswith("image_fovea"):
            cache[key] = l2_normalize(value.float())
    return cache


def split_feature_indices(cache: dict, split: str) -> slice:
    a, b = cache["split_ranges"][split]
    return slice(int(a), int(b))


def features_for_ids(cache: dict, image_ids: Sequence[str], key: str) -> torch.Tensor:
    lookup = {image_id: i for i, image_id in enumerate(cache["image_ids"])}
    idx = [lookup[i] for i in image_ids]
    return cache[key][idx]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", help="Sub-command: 'single' for legacy single-backbone, 'multi' for multi-backend.")
    single = sub.add_parser("single", help="Legacy single-CLIP-backend feature extraction")
    single.add_argument("--data-dir", default="image-eeg-data")
    single.add_argument("--image-root", default="auto")
    single.add_argument("--clip-backbone", default="RN50")
    single.add_argument("--clip-pretrained", default="openai")
    single.add_argument("--output-cache", default="cache/features_rn50.pt")
    single.add_argument("--batch-size", type=int, default=64)
    single.add_argument("--feature-dim", type=int, default=512)
    single.add_argument("--sigma0", type=float, default=8.0)
    single.add_argument("--blur-c", type=float, default=6.0)
    single.add_argument("--device", default="auto")
    single.add_argument("--clean-only", action="store_true")
    multi = sub.add_parser("multi", help="Multi-backend feature extraction (RN50 + ViT-B/32 + DINOv2 + VAE)")
    multi.add_argument("--data-dir", default="image-eeg-data")
    multi.add_argument("--image-root", default="auto")
    multi.add_argument("--backends", nargs="+", default=["RN50", "ViT-B-32", "dinov2_da2", "sd_vae"],
                       help="List of backends: RN50, ViT-B-32, ViT-L-14, dinov2, dinov2_da2, sd_vae")
    multi.add_argument("--dinov2-model", default="dinov2_vitb14", help="DINOv2 model name for torch.hub")
    multi.add_argument("--vae-name", default="stabilityai/sd-vae-ft-mse", help="VAE model name for diffusers")
    multi.add_argument("--output-cache", default="cache/features_multi.pt")
    multi.add_argument("--batch-size", type=int, default=64)
    multi.add_argument("--feature-dim", type=int, default=512)
    multi.add_argument("--device", default="auto")
    p.add_argument("--data-dir", default="image-eeg-data")
    p.add_argument("--image-root", default="auto")
    p.add_argument("--clip-backbone", default="RN50")
    p.add_argument("--clip-pretrained", default="openai")
    p.add_argument("--output-cache", default="cache/features_rn50.pt")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--feature-dim", type=int, default=512)
    p.add_argument("--sigma0", type=float, default=8.0)
    p.add_argument("--blur-c", type=float, default=6.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--clean-only", action="store_true")
    p.add_argument("--mode", choices=["single", "multi"], default=None,
                   help="Legacy mode selection. Use 'single' for old CLIP-only, 'multi' for multi-backend. If not set, defaults to single for backward compat.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mode = args.mode or (args.command or "single")
    if mode == "multi" or args.command == "multi":
        build_multi_feature_cache(
            args.data_dir,
            getattr(args, "image_root", "auto"),
            args.output_cache,
            backends=args.backends,
            dinov2_model=args.dinov2_model,
            vae_name=args.vae_name,
            batch_size=args.batch_size,
            feature_dim=args.feature_dim,
            device=args.device,
        )
    else:
        build_feature_cache(
            args.data_dir,
            args.image_root,
            args.clip_backbone,
            args.output_cache,
            clip_pretrained=args.clip_pretrained,
            batch_size=args.batch_size,
            feature_dim=args.feature_dim,
            sigma0=args.sigma0,
            blur_c=args.blur_c,
            device=args.device,
            clean_only=args.clean_only,
        )


if __name__ == "__main__":
    main()
