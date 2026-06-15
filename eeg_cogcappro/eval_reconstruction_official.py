from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import scipy as sp
import torch
from PIL import Image
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
from torchvision import transforms
from torchvision.models import AlexNet_Weights, EfficientNet_B1_Weights, Inception_V3_Weights, alexnet, efficientnet_b1, inception_v3
from torchvision.models.feature_extraction import create_feature_extractor

from .data import load_eeg_dataset
from .utils import choose_device, ensure_dir, write_json


def _local_open_clip_checkpoint() -> str | None:
    root = Path.home() / ".cache/huggingface/hub/models--laion--CLIP-ViT-L-14-laion2B-s32B-b82K/blobs"
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_file() and not p.name.endswith(".incomplete")]
    if not candidates:
        return None
    return str(max(candidates, key=lambda p: p.stat().st_size))


def load_stack(paths: list[Path], size: int = 256) -> torch.Tensor:
    imgs = []
    for path in paths:
        with Image.open(path) as img:
            arr = np.asarray(img.convert("RGB").resize((size, size)), dtype=np.float32) / 255.0
        imgs.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(imgs)


def find_real_paths(data_dir: str | Path, real_dir: str | Path | None, n: int) -> list[Path]:
    if real_dir:
        root = Path(real_dir)
        paths = sorted(root.rglob("*.jpg")) + sorted(root.rglob("*.jpeg")) + sorted(root.rglob("*.png"))
        return [p for p in paths if not p.name.startswith("._")][:n]
    split = load_eeg_dataset(data_dir, "test", avg_trials=True, image_root="auto")
    return [p for p in split.image_paths if p is not None][:n]


def fallback_eval(real: torch.Tensor, fake: torch.Tensor) -> dict[str, float]:
    real = real.float()
    fake = fake.float()
    mse = torch.mean((real - fake) ** 2).item()
    cos = torch.nn.functional.cosine_similarity(real.flatten(1), fake.flatten(1), dim=1).mean().item()
    return {"eval_mse_fallback": float(mse), "eval_pixel_cosine_fallback": float(cos)}


def two_way_identification(
    recons: torch.Tensor,
    images: torch.Tensor,
    model: Callable,
    preprocess: Callable,
    feature_layer: str | None = None,
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    pred_parts = []
    real_parts = []
    for start in range(0, len(images), batch_size):
        pred = model(preprocess(recons[start : start + batch_size]).to(device))
        real = model(preprocess(images[start : start + batch_size]).to(device))
        if feature_layer is not None:
            pred = pred[feature_layer]
            real = real[feature_layer]
        pred_parts.append(pred.float().flatten(1).detach().cpu())
        real_parts.append(real.float().flatten(1).detach().cpu())
    preds = torch.cat(pred_parts).numpy()
    reals = torch.cat(real_parts).numpy()
    r = np.corrcoef(reals, preds)[: len(images), len(images) :]
    congruents = np.diag(r)
    success_cnt = np.sum(r < congruents, axis=0)
    return float(np.mean(success_cnt / (len(images) - 1)))


def metric_pixcorr(real: torch.Tensor, fake: torch.Tensor, *_args, **_kwargs) -> float:
    preprocess = transforms.Compose([transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR)])
    real_flat = preprocess(real).reshape(len(real), -1).cpu()
    fake_flat = preprocess(fake).reshape(len(fake), -1).cpu()
    values = [np.corrcoef(real_flat[i], fake_flat[i])[0][1] for i in range(min(len(real_flat), len(fake_flat)))]
    return float(np.mean(values))


def metric_ssim(real: torch.Tensor, fake: torch.Tensor, *_args, **_kwargs) -> float:
    preprocess = transforms.Compose([transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR)])
    real_gray = rgb2gray(preprocess(real).permute((0, 2, 3, 1)).cpu())
    fake_gray = rgb2gray(preprocess(fake).permute((0, 2, 3, 1)).cpu())
    values = [
        structural_similarity(
            fake_gray[i],
            real_gray[i],
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            data_range=1.0,
        )
        for i in range(len(real_gray))
    ]
    return float(np.mean(values))


def metric_alexnet(real: torch.Tensor, fake: torch.Tensor, device: torch.device, batch_size: int) -> tuple[float, float]:
    weights = AlexNet_Weights.IMAGENET1K_V1
    model = create_feature_extractor(alexnet(weights=weights), return_nodes=["features.4", "features.11"]).to(device)
    model.eval().requires_grad_(False)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    alex2 = two_way_identification(fake.float(), real, model, preprocess, "features.4", device=device, batch_size=batch_size)
    alex5 = two_way_identification(fake.float(), real, model, preprocess, "features.11", device=device, batch_size=batch_size)
    return alex2, alex5


def metric_inception(real: torch.Tensor, fake: torch.Tensor, device: torch.device, batch_size: int) -> float:
    weights = Inception_V3_Weights.DEFAULT
    model = create_feature_extractor(inception_v3(weights=weights), return_nodes=["avgpool"]).to(device)
    model.eval().requires_grad_(False)
    preprocess = transforms.Compose(
        [
            transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return two_way_identification(fake, real, model, preprocess, "avgpool", device=device, batch_size=batch_size)


def metric_clip(real: torch.Tensor, fake: torch.Tensor, device: torch.device, batch_size: int, allow_open_clip_fallback: bool) -> tuple[float, str]:
    preprocess = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ]
    )
    try:
        import clip

        model, _ = clip.load("ViT-L/14", device=device)
        model.eval()
        return (
            two_way_identification(fake, real, model.encode_image, preprocess, None, device=device, batch_size=batch_size),
            "official_openai_clip_vit_l_14",
        )
    except Exception as exc:
        if not allow_open_clip_fallback:
            raise
        try:
            import open_clip

            local_ckpt = _local_open_clip_checkpoint()
            pretrained = local_ckpt or "laion2b_s32b_b82k"
            model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained=pretrained)
            model = model.eval().to(device)
            return (
                two_way_identification(fake, real, model.encode_image, preprocess, None, device=device, batch_size=batch_size),
                f"open_clip_vit_l_14_laion2b_s32b_b82k_fallback_after_openai_clip_error: {exc}; pretrained={pretrained}",
            )
        except Exception as fallback_exc:
            raise RuntimeError(f"OpenAI clip failed: {exc}; open_clip fallback failed: {fallback_exc}") from fallback_exc


def metric_effnet(real: torch.Tensor, fake: torch.Tensor, device: torch.device, batch_size: int) -> float:
    weights = EfficientNet_B1_Weights.DEFAULT
    model = create_feature_extractor(efficientnet_b1(weights=weights), return_nodes=["avgpool"]).to(device)
    model.eval().requires_grad_(False)
    preprocess = transforms.Compose(
        [
            transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    real_parts = []
    fake_parts = []
    for start in range(0, len(real), batch_size):
        real_parts.append(model(preprocess(real[start : start + batch_size]).to(device))["avgpool"].float().reshape(-1, 1280).detach().cpu())
        fake_parts.append(model(preprocess(fake[start : start + batch_size]).to(device))["avgpool"].float().reshape(-1, 1280).detach().cpu())
    gt = torch.cat(real_parts).numpy()
    pred = torch.cat(fake_parts).numpy()
    return float(np.array([sp.spatial.distance.correlation(gt[i], pred[i]) for i in range(len(gt))]).mean())


def metric_swav(real: torch.Tensor, fake: torch.Tensor, device: torch.device, batch_size: int) -> float:
    model = torch.hub.load("facebookresearch/swav:main", "resnet50")
    model = create_feature_extractor(model, return_nodes=["avgpool"]).to(device)
    model.eval().requires_grad_(False)
    preprocess = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    real_parts = []
    fake_parts = []
    for start in range(0, len(real), batch_size):
        real_parts.append(model(preprocess(real[start : start + batch_size]).to(device))["avgpool"].float().flatten(1).detach().cpu())
        fake_parts.append(model(preprocess(fake[start : start + batch_size]).to(device))["avgpool"].float().flatten(1).detach().cpu())
    gt = torch.cat(real_parts).numpy()
    pred = torch.cat(fake_parts).numpy()
    return float(np.array([sp.spatial.distance.correlation(gt[i], pred[i]) for i in range(len(gt))]).mean())


@torch.no_grad()
def eval_images(
    real: torch.Tensor,
    fake: torch.Tensor,
    *,
    device: torch.device,
    metrics: str = "all",
    batch_size: int = 32,
    allow_open_clip_fallback: bool = False,
) -> dict:
    real = real.to(device).float()
    fake = fake.to(device).float()
    out: dict[str, object] = {"num_images": int(len(real)), "device": str(device)}
    errors: dict[str, str] = {}

    requested = {"pixcorr", "ssim", "alexnet", "inception", "clip", "effnet", "swav"} if metrics == "all" else {"ssim", "alexnet", "clip"}

    def capture(name: str, fn: Callable[[], object]) -> object | None:
        try:
            return fn()
        except Exception as exc:
            errors[name] = str(exc)
            return None

    if "pixcorr" in requested:
        value = capture("pixcorr", lambda: metric_pixcorr(real, fake))
        if value is not None:
            out["eval_pixcorr"] = value
    if "ssim" in requested:
        value = capture("ssim", lambda: metric_ssim(real, fake))
        if value is not None:
            out["eval_ssim"] = value
    if "alexnet" in requested:
        value = capture("alexnet", lambda: metric_alexnet(real, fake, device, batch_size))
        if value is not None:
            out["eval_alex2"], out["eval_alex5"] = value
    if "inception" in requested:
        value = capture("inception", lambda: metric_inception(real, fake, device, batch_size))
        if value is not None:
            out["eval_inception"] = value
    if "clip" in requested:
        value = capture("clip", lambda: metric_clip(real, fake, device, batch_size, allow_open_clip_fallback))
        if value is not None:
            out["eval_clip"], out["clip_backend"] = value
    if "effnet" in requested:
        value = capture("effnet", lambda: metric_effnet(real, fake, device, batch_size))
        if value is not None:
            out["eval_effnet"] = value
    if "swav" in requested:
        value = capture("swav", lambda: metric_swav(real, fake, device, batch_size))
        if value is not None:
            out["eval_swav"] = value
    if errors:
        out["metric_errors"] = errors
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Official-compatible reconstruction evaluation.")
    p.add_argument("--real-dir", default=None)
    p.add_argument("--fake-dir", required=True)
    p.add_argument("--data-dir", default="image-eeg-data")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--metrics", choices=["all", "requested"], default="all")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--allow-open-clip-fallback", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fake_paths = sorted(Path(args.fake_dir).glob("*.png"))
    real_paths = find_real_paths(args.data_dir, args.real_dir, len(fake_paths))
    if len(real_paths) < len(fake_paths):
        raise FileNotFoundError("Could not locate enough real test images for reconstruction evaluation.")
    real = load_stack(real_paths[: len(fake_paths)])
    fake = load_stack(fake_paths)
    device = choose_device(args.device)
    metrics = eval_images(
        real,
        fake,
        device=device,
        metrics=args.metrics,
        batch_size=args.batch_size,
        allow_open_clip_fallback=args.allow_open_clip_fallback,
    )
    fallback = fallback_eval(real, fake)
    metrics.update(fallback)
    ensure_dir(Path(args.output).parent)
    write_json(args.output, metrics)
    print(f"Wrote reconstruction metrics: {args.output}", flush=True)


if __name__ == "__main__":
    main()
