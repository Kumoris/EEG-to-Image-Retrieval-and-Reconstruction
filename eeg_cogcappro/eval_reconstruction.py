from __future__ import annotations

import argparse
from pathlib import Path

from .eval_reconstruction_official import eval_images, fallback_eval, find_real_paths, load_stack
from .utils import choose_device, ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--real-dir", default=None)
    p.add_argument("--fake-dir", required=True)
    p.add_argument("--data-dir", default="image-eeg-data")
    p.add_argument("--output", default="results/recon_seed0.json")
    p.add_argument("--device", default="auto")
    p.add_argument("--metrics", choices=["all", "requested"], default="all")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--allow-open-clip-fallback", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fake_dir = Path(args.fake_dir)
    fake_paths = sorted(fake_dir.glob("*.png"))
    real_paths = find_real_paths(args.data_dir, args.real_dir, len(fake_paths))
    if len(real_paths) < len(fake_paths):
        raise FileNotFoundError("Could not locate enough test real images. Provide --real-dir or ensure image_resolver can locate test image_id.")
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
    metrics.update(fallback_eval(real, fake))
    metrics["note"] = "official-compatible project eval; per-metric errors are recorded in metric_errors"
    ensure_dir(Path(args.output).parent)
    write_json(args.output, metrics)
    print(f"Wrote reconstruction metrics: {args.output}", flush=True)


if __name__ == "__main__":
    main()
