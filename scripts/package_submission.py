#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from eeg_cogcappro.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package baseline outputs into one zip file.")
    parser.add_argument("--output-dir", default="outputs/baseline")
    parser.add_argument("--zip-path", default="outputs/baseline/submission.zip")
    parser.add_argument("--retrieval-dir", default=None, help="Directory containing retrieval outputs to copy into output-dir.")
    parser.add_argument("--recon-dir", default=None, help="Directory containing reconstruction PNGs plus manifest.csv/summary.json.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--expected-recons", type=int, default=200)
    return parser.parse_args()


def _copy_file(src: Path, dst: Path) -> Path:
    ensure_dir(dst.parent)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return dst


def _stage_retrieval(output_dir: Path, retrieval_dir: Path, split: str) -> list[Path]:
    names = [
        f"retrieval_{split}_metrics.json",
        f"retrieval_{split}_logits.pt",
        f"retrieval_{split}_top5.csv",
        f"retrieval_{split}_rankings.csv",
        f"retrieval_{split}_rankings.json",
    ]
    staged = []
    for name in names:
        src = retrieval_dir / name
        if src.exists():
            staged.append(_copy_file(src, output_dir / name))
    required = [output_dir / f"retrieval_{split}_metrics.json", output_dir / f"retrieval_{split}_logits.pt"]
    if not any((output_dir / name).exists() for name in [f"retrieval_{split}_top5.csv", f"retrieval_{split}_rankings.csv"]):
        required.append(output_dir / f"retrieval_{split}_top5.csv")
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required retrieval output(s): " + ", ".join(str(p) for p in missing))
    return sorted({p for p in staged if p.exists()} | {p for p in output_dir.glob(f"retrieval_{split}_*") if p.is_file()})


def _stage_reconstructions(output_dir: Path, recon_source: Path, expected: int) -> tuple[Path, Path, Path | None, list[Path]]:
    recon_dir = ensure_dir(output_dir / "reconstructions")
    pngs = sorted(recon_source.glob("*.png"))
    if len(pngs) != expected:
        raise FileNotFoundError(f"Expected {expected} reconstruction PNGs in {recon_source}, found {len(pngs)}")
    staged_pngs = [_copy_file(path, recon_dir / path.name) for path in pngs]

    manifest_src = recon_source / "manifest.csv"
    if not manifest_src.exists():
        manifest_src = output_dir / "reconstruction_manifest.csv"
    if not manifest_src.exists():
        raise FileNotFoundError(f"Missing reconstruction manifest: {recon_source / 'manifest.csv'}")
    manifest = _copy_file(manifest_src, output_dir / "reconstruction_manifest.csv")

    summary = None
    summary_src = recon_source / "summary.json"
    if summary_src.exists():
        summary = _copy_file(summary_src, output_dir / "reconstruction_summary.json")
    elif (output_dir / "reconstruction_summary.json").exists():
        summary = output_dir / "reconstruction_summary.json"
    return recon_dir, manifest, summary, staged_pngs


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    zip_path = Path(args.zip_path)
    ensure_dir(output_dir)
    ensure_dir(zip_path.parent)
    retrieval_dir = Path(args.retrieval_dir) if args.retrieval_dir else output_dir
    recon_source = Path(args.recon_dir) if args.recon_dir else output_dir / "reconstructions"

    retrieval_files = _stage_retrieval(output_dir, retrieval_dir, args.split)
    recon_dir, manifest, summary, staged_pngs = _stage_reconstructions(output_dir, recon_source, args.expected_recons)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in retrieval_files:
            zf.write(path, arcname=path.name)
        zf.write(manifest, arcname=manifest.name)
        if summary is not None:
            zf.write(summary, arcname=summary.name)
        for path in staged_pngs:
            zf.write(path, arcname=f"reconstructions/{path.name}")
    print(f"Wrote package: {zip_path}")
    print(f"Included {len(retrieval_files)} retrieval file(s), {len(staged_pngs)} reconstruction PNGs, manifest {manifest.name}.")


if __name__ == "__main__":
    main()
