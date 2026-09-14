# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convert paired Labelme circle annotations to deterministic YOLO segmentation polygons."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

from PIL import Image


DATASET_ROOT = Path(
    "/Users/huangxiaofang/Documents/computer_technology/algorithm-cv/project/practice/datasets/pt/PT缺陷/PT缺陷/yolo_seg"
)
CLASS_IDS = {"out": 0, "in": 1}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_ROOT, help="Root containing images/ and labelme/")
    parser.add_argument("--points", type=int, default=72, help="Number of polygon vertices sampled around each circle")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing labels directory")
    return parser.parse_args()


def image_by_stem(directory: Path) -> dict[str, Path]:
    images = {
        path.stem: path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    if not images:
        raise ValueError(f"No images found in {directory}")
    return images


def circle_polygon(shape: dict, width: int, height: int, point_count: int) -> list[float]:
    if shape.get("shape_type") != "circle":
        raise ValueError(f"Expected circle, got {shape.get('shape_type')!r}")
    points = shape.get("points")
    if not isinstance(points, list) or len(points) != 2:
        raise ValueError("A Labelme circle must contain exactly two points")
    (cx, cy), (px, py) = points
    cx, cy, px, py = map(float, (cx, cy, px, py))
    radius = math.hypot(px - cx, py - cy)
    if radius <= 0:
        raise ValueError("Circle radius must be positive")
    if cx - radius < 0 or cy - radius < 0 or cx + radius > width or cy + radius > height:
        raise ValueError("Circle extends beyond the image boundary")

    polygon: list[float] = []
    for index in range(point_count):
        angle = 2 * math.pi * index / point_count
        x = (cx + radius * math.cos(angle)) / width
        y = (cy + radius * math.sin(angle)) / height
        polygon.extend((x, y))
    return polygon


def convert_split(dataset: Path, split: str, output_root: Path, point_count: int) -> dict:
    image_dir = dataset / "images" / split
    labelme_dir = dataset / "labelme" / split
    output_dir = output_root / split
    images = image_by_stem(image_dir)
    json_files = {path.stem: path for path in labelme_dir.glob("*.json") if path.is_file()}
    if images.keys() != json_files.keys():
        raise ValueError(
            f"{split} image/JSON mismatch: missing_json={sorted(images.keys() - json_files.keys())}, "
            f"missing_image={sorted(json_files.keys() - images.keys())}"
        )

    class_counts = {label: 0 for label in CLASS_IDS}
    for stem in sorted(images):
        image_path = images[stem]
        json_path = json_files[stem]
        data = json.loads(json_path.read_text(encoding="utf-8"))
        with Image.open(image_path) as image:
            width, height = image.size
        if data.get("imageWidth") != width or data.get("imageHeight") != height:
            raise ValueError(f"{json_path.name}: declared dimensions do not match {image_path.name}")
        if Path(str(data.get("imagePath", ""))).stem != stem:
            raise ValueError(f"{json_path.name}: imagePath does not match the JSON filename")

        shapes = data.get("shapes")
        if not isinstance(shapes, list):
            raise ValueError(f"{json_path.name}: shapes must be a list")
        grouped: dict[str, list[dict]] = {label: [] for label in CLASS_IDS}
        for shape in shapes:
            label = shape.get("label")
            if label not in CLASS_IDS:
                raise ValueError(f"{json_path.name}: unexpected class {label!r}")
            grouped[label].append(shape)
        if any(len(grouped[label]) != 1 for label in CLASS_IDS):
            counts = {label: len(grouped[label]) for label in CLASS_IDS}
            raise ValueError(f"{json_path.name}: expected one out and one in circle, got {counts}")

        lines = []
        for label, class_id in CLASS_IDS.items():
            polygon = circle_polygon(grouped[label][0], width, height, point_count)
            lines.append(" ".join([str(class_id), *(f"{coordinate:.6f}" for coordinate in polygon)]))
            class_counts[label] += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"images": len(images), "labels": len(json_files), "class_counts": class_counts}


def main() -> None:
    args = parse_args()
    if not 12 <= args.points <= 720:
        raise ValueError("points must be between 12 and 720")
    dataset = args.dataset.resolve()
    labels_dir = dataset / "labels"
    temporary_dir = dataset / ".labels-converting"
    if labels_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Labels already exist: {labels_dir}. Pass --overwrite to replace them.")
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)

    try:
        result = {
            split: convert_split(dataset, split, temporary_dir, args.points) for split in ("train", "val")
        }
        if labels_dir.exists():
            shutil.rmtree(labels_dir)
        temporary_dir.rename(labels_dir)
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise

    report = {
        "classes": CLASS_IDS,
        "points_per_circle": args.points,
        "splits": result,
    }
    report_path = dataset / "conversion_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
