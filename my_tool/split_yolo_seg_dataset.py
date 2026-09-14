# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Reproducibly split and move paired PT images and Labelme JSON files."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime
from pathlib import Path


DATASET_ROOT = Path(
    "/Users/huangxiaofang/Documents/computer_technology/algorithm-cv/project/practice/datasets/pt/PT缺陷/PT缺陷"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=DATASET_ROOT / "seg")
    parser.add_argument("--labelme", type=Path, default=DATASET_ROOT / "seg_labels")
    parser.add_argument("--output", type=Path, default=DATASET_ROOT / "yolo_seg")
    parser.add_argument("--train-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def paired_files(image_dir: Path, label_dir: Path) -> dict[str, tuple[Path, Path]]:
    """Return image/JSON pairs after strict one-to-one validation."""
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise ValueError(f"Source directory missing: images={image_dir}, labelme={label_dir}")

    images = {path.stem: path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES}
    labels = {path.stem: path for path in label_dir.glob("*.json") if path.is_file()}
    missing_labels = sorted(images.keys() - labels.keys())
    missing_images = sorted(labels.keys() - images.keys())
    if missing_labels or missing_images:
        raise ValueError(f"Unpaired files: missing_labels={missing_labels}, missing_images={missing_images}")
    if not images:
        raise ValueError("No paired source files found")
    return {stem: (images[stem], labels[stem]) for stem in sorted(images)}


def destination_paths(output: Path, split: str, image: Path, label: Path) -> tuple[Path, Path]:
    return output / "images" / split / image.name, output / "labelme" / split / label.name


def main() -> None:
    args = parse_args()
    pairs = paired_files(args.images.resolve(), args.labelme.resolve())
    if not 0 < args.train_count < len(pairs):
        raise ValueError(f"train-count must be between 1 and {len(pairs) - 1}")

    stems = list(pairs)
    random.Random(args.seed).shuffle(stems)
    train_stems = stems[: args.train_count]
    val_stems = stems[args.train_count :]
    split_by_stem = {stem: "train" for stem in train_stems} | {stem: "val" for stem in val_stems}

    operations: list[tuple[Path, Path]] = []
    for stem, (image, label) in pairs.items():
        image_dst, label_dst = destination_paths(args.output.resolve(), split_by_stem[stem], image, label)
        operations.extend(((image, image_dst), (label, label_dst)))
    existing = [str(destination) for _, destination in operations if destination.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing destination files: {existing[:5]}")

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": args.seed,
        "source_images": str(args.images.resolve()),
        "source_labelme": str(args.labelme.resolve()),
        "output": str(args.output.resolve()),
        "train": train_stems,
        "val": val_stems,
    }
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    moved: list[tuple[Path, Path]] = []
    try:
        for source, destination in operations:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(source, destination)
            moved.append((source, destination))
        manifest_path = args.output.resolve() / "split_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        for source, destination in reversed(moved):
            if destination.exists() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(destination, source)
        raise

    print(f"Moved {len(train_stems)} train and {len(val_stems)} val image/JSON pairs to {args.output.resolve()}")


if __name__ == "__main__":
    main()
