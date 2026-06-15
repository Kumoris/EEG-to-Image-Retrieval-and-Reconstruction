#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create tiny synthetic Project1-like data for smoke tests.")
    parser.add_argument("--data-dir", default="/tmp/brain2image_toy")
    parser.add_argument("--train-items", type=int, default=24)
    parser.add_argument("--test-items", type=int, default=8)
    parser.add_argument("--channels", type=int, default=63)
    parser.add_argument("--time-steps", type=int, default=250)
    return parser.parse_args()


def write_images(data_dir: Path, total: int) -> list[str]:
    from PIL import Image, ImageDraw

    image_dir = data_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for i in range(total):
        name = f"toy_{i:03d}.png"
        path = image_dir / name
        color = ((37 * i) % 255, (83 * i) % 255, (149 * i) % 255)
        img = Image.new("RGB", (128, 128), color=color)
        draw = ImageDraw.Draw(img)
        draw.text((12, 54), f"{i:03d}", fill=(255, 255, 255))
        img.save(path)
        names.append(str(path))
    return names


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    names = write_images(data_dir, args.train_items + args.test_items)

    generator = torch.Generator().manual_seed(0)
    train_eeg = torch.randn(args.train_items, 4, args.channels, args.time_steps, generator=generator)
    test_eeg = torch.randn(args.test_items, 4, args.channels, args.time_steps, generator=generator)
    torch.save({"eeg": train_eeg, "img": np.array(names[: args.train_items]).reshape(-1, 1)}, data_dir / "train.pt")
    torch.save({"eeg": test_eeg, "img": np.array(names[args.train_items :]).reshape(-1, 1)}, data_dir / "test.pt")

    with (data_dir / "EEG_CHANNELS.jsonl").open("w", encoding="utf-8") as f:
        for i in range(args.channels):
            f.write(json.dumps({"name": f"CH{i:02d}"}) + "\n")
    print(f"Wrote toy data: {data_dir}")


if __name__ == "__main__":
    main()
