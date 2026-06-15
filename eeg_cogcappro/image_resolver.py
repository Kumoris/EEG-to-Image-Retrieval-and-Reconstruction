from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and not path.name.startswith("._")


def candidate_roots(data_dir: str | Path, image_root: str | Path | Sequence[str | Path] | None = "auto") -> list[Path]:
    data_dir = Path(data_dir)
    roots: list[Path] = []
    if image_root and image_root != "auto":
        if isinstance(image_root, (str, Path)):
            roots.append(Path(image_root))
        else:
            roots.extend(Path(p) for p in image_root)
    search_bases = [data_dir, Path.cwd(), Path.cwd().parent]
    names = ["images", "image", "stimuli", "stimulus_images", "THINGS", "things", "image_set", "training_images", "test_images"]
    for base in search_bases:
        roots.append(base)
        for name in names:
            roots.append(base / name)
    seen: set[Path] = set()
    out: list[Path] = []
    for root in roots:
        try:
            resolved = root.expanduser().resolve()
        except OSError:
            continue
        if resolved.exists() and resolved not in seen and "__MACOSX" not in resolved.parts:
            out.append(resolved)
            seen.add(resolved)
    return out


def build_image_index(roots: Iterable[str | Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root_like in roots:
        root = Path(root_like)
        if _is_image(root):
            index.setdefault(root.stem, root)
            index.setdefault(root.name, root)
            continue
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if "__MACOSX" in path.parts or path.name.startswith("._"):
                continue
            if _is_image(path):
                index.setdefault(path.stem, path)
                index.setdefault(path.name, path)
    return index


class ImageResolver:
    def __init__(self, data_dir: str | Path, image_root: str | Path | Sequence[str | Path] | None = "auto") -> None:
        self.data_dir = Path(data_dir)
        self.roots = candidate_roots(data_dir, image_root)
        self.index = build_image_index(self.roots)

    def resolve(self, image_id_or_path: str | Path) -> Optional[Path]:
        raw = Path(str(image_id_or_path))
        direct = [raw, self.data_dir / raw, self.data_dir / raw.name]
        for path in direct:
            if _is_image(path):
                return path
        return self.index.get(raw.stem) or self.index.get(raw.name)

    def resolve_many(self, image_ids: Sequence[str | Path], *, warn: bool = True) -> list[Optional[Path]]:
        paths = [self.resolve(x) for x in image_ids]
        missing = [str(x) for x, p in zip(image_ids, paths) if p is None]
        if warn and missing:
            preview = ", ".join(missing[:20])
            suffix = " ..." if len(missing) > 20 else ""
            print(f"Warning: missing {len(missing)}/{len(image_ids)} images: {preview}{suffix}", flush=True)
        return paths


def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")
