#!/usr/bin/env python3
"""Select high-quality YOLO segmentation pseudo-labels and their source images."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter
from pathlib import Path

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Directory containing the source images.")
    parser.add_argument("--labels", type=Path, required=True, help="Directory containing predicted YOLO-seg labels.")
    parser.add_argument("--output", type=Path, required=True, help="Output root; images/ and labels/ are created here.")
    parser.add_argument("--min-confidence", type=float, default=0.965, help="Minimum confidence for both instances.")
    parser.add_argument("--min-area-ratio", type=float, default=0.45, help="Minimum inner/outer polygon area ratio.")
    parser.add_argument("--max-area-ratio", type=float, default=0.60, help="Maximum inner/outer polygon area ratio.")
    parser.add_argument(
        "--max-center-offset", type=float, default=0.05, help="Maximum center offset / outer equivalent radius."
    )
    parser.add_argument(
        "--min-containment", type=float, default=0.99, help="Minimum fraction of inner points inside outer polygon."
    )
    parser.add_argument("--dry-run", action="store_true", help="Analyze without copying images or labels.")
    return parser.parse_args()


def polygon_area(points: list[tuple[float, float]]) -> float:
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]))) / 2


def polygon_center(points: list[tuple[float, float]]) -> tuple[float, float]:
    return sum(x for x, _ in points) / len(points), sum(y for _, y in points) / len(points)


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def parse_label(path: Path) -> list[dict]:
    instances = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        tokens = line.split()
        if not tokens:
            continue
        if len(tokens) < 8 or len(tokens) % 2:
            raise ValueError(f"line {line_number}: expected class, polygon coordinate pairs, and confidence")
        coordinates = tokens[1:-1]
        points = list(zip(map(float, coordinates[::2]), map(float, coordinates[1::2])))
        instances.append(
            {
                "class_id": int(tokens[0]),
                "confidence": float(tokens[-1]),
                "points": points,
                "training_line": " ".join(tokens[:-1]),
            }
        )
    return instances


def find_image(images: dict[str, Path], stem: str) -> Path | None:
    return images.get(stem)


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def main() -> None:
    args = parse_args()
    image_paths = [p for p in args.images.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    images = {p.stem: p for p in image_paths}
    if len(images) != len(image_paths):
        raise RuntimeError("Source image stems are not unique; labels cannot be matched safely.")

    label_paths = {p.stem: p for p in args.labels.glob("*.txt")}
    records = []
    rejection_counts = Counter()
    for stem in sorted(images.keys() | label_paths.keys()):
        image_path = find_image(images, stem)
        label_path = label_paths.get(stem)
        if label_path is None:
            records.append(
                {
                    "stem": stem,
                    "image": str(image_path),
                    "label": "",
                    "selected": False,
                    "reasons": "missing_label",
                    "min_confidence": None,
                    "area_ratio": None,
                    "center_offset": None,
                    "containment": None,
                }
            )
            rejection_counts["missing_label"] += 1
            continue
        reasons = []
        try:
            instances = parse_label(label_path)
        except (TypeError, ValueError) as error:
            records.append(
                {
                    "stem": stem,
                    "image": str(image_path) if image_path else "",
                    "label": str(label_path),
                    "selected": False,
                    "reasons": f"invalid_label:{error}",
                    "min_confidence": None,
                    "area_ratio": None,
                    "center_offset": None,
                    "containment": None,
                }
            )
            rejection_counts["invalid_label"] += 1
            continue

        classes = Counter(instance["class_id"] for instance in instances)
        if len(instances) != 2 or classes != Counter({0: 1, 1: 1}):
            reasons.append("expected_one_out_and_one_in")

        if image_path is None:
            reasons.append("missing_image")

        metrics = {"min_confidence": None, "area_ratio": None, "center_offset": None, "containment": None}
        if not reasons:
            by_class = {instance["class_id"]: instance for instance in instances}
            outer, inner = by_class[0], by_class[1]
            outer_area = polygon_area(outer["points"])
            inner_area = polygon_area(inner["points"])
            if outer_area <= 0 or inner_area <= 0:
                reasons.append("non_positive_area")
            else:
                outer_center = polygon_center(outer["points"])
                inner_center = polygon_center(inner["points"])
                metrics = {
                    "min_confidence": min(outer["confidence"], inner["confidence"]),
                    "area_ratio": inner_area / outer_area,
                    "center_offset": math.dist(outer_center, inner_center) / math.sqrt(outer_area / math.pi),
                    "containment": sum(point_in_polygon(point, outer["points"]) for point in inner["points"])
                    / len(inner["points"]),
                }
                if metrics["min_confidence"] < args.min_confidence:
                    reasons.append("low_confidence")
                if not args.min_area_ratio <= metrics["area_ratio"] <= args.max_area_ratio:
                    reasons.append("area_ratio")
                if metrics["center_offset"] > args.max_center_offset:
                    reasons.append("center_offset")
                if metrics["containment"] < args.min_containment:
                    reasons.append("containment")

        for reason in reasons:
            rejection_counts[reason] += 1
        records.append(
            {
                "stem": stem,
                "image": str(image_path) if image_path else "",
                "label": str(label_path),
                "selected": not reasons,
                "reasons": ";".join(reasons),
                **metrics,
            }
        )

    selected = [record for record in records if record["selected"]]
    if not args.dry_run:
        output_images = args.output / "images"
        output_labels = args.output / "labels"
        if any(output_images.glob("*")) or any(output_labels.glob("*")):
            raise RuntimeError(f"Output images/ or labels/ is not empty: {args.output}")
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        for record in selected:
            image_path = Path(record["image"])
            shutil.copy2(image_path, output_images / image_path.name)
            instances = parse_label(Path(record["label"]))
            (output_labels / f"{record['stem']}.txt").write_text(
                "\n".join(instance["training_line"] for instance in instances) + "\n"
            )

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "selection_manifest.csv"
    with manifest_path.open("w", newline="") as file:
        fields = [
            "stem",
            "image",
            "label",
            "selected",
            "reasons",
            "min_confidence",
            "area_ratio",
            "center_offset",
            "containment",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    valid_confidences = [record["min_confidence"] for record in records if record["min_confidence"] is not None]
    report = {
        "input_images": len(image_paths),
        "input_labels": len(label_paths),
        "selected": len(selected),
        "rejected": len(records) - len(selected),
        "thresholds": {
            "required_classes": {"0_out": 1, "1_in": 1},
            "min_confidence": args.min_confidence,
            "area_ratio": [args.min_area_ratio, args.max_area_ratio],
            "max_center_offset": args.max_center_offset,
            "min_containment": args.min_containment,
        },
        "min_confidence_distribution": {
            "min": min(valid_confidences),
            "p25": quantile(valid_confidences, 0.25),
            "median": quantile(valid_confidences, 0.5),
            "p75": quantile(valid_confidences, 0.75),
            "max": max(valid_confidences),
        },
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "confidence_removed_from_training_labels": True,
        "dry_run": args.dry_run,
    }
    (args.output / "selection_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
