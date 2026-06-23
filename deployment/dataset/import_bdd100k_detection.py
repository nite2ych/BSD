"""
Import official BDD100K detection labels into BSD YOLO format.

Expected BDD100K layout:
  <bdd-root>/
    images/100k/train/*.jpg
    images/100k/val/*.jpg
    labels/det_20/det_train.json
    labels/det_20/det_val.json

The script maps BDD labels to BSD's four classes:
  person, bicycle, motorcycle, vehicle

It writes:
  <output>/<split>/images/
  <output>/<split>/labels/
  <output>/dataset.yaml
  <output>/manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BSD_NAMES = ["person", "bicycle", "motorcycle", "vehicle"]
IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png"]

BDD_TO_BSD = {
    "person": 0,
    "pedestrian": 0,
    "rider": 0,
    "bicycle": 1,
    "bike": 1,
    "motorcycle": 2,
    "motorbike": 2,
    "scooter": 2,
    "car": 3,
    "bus": 3,
    "truck": 3,
    "van": 3,
}


@dataclass
class ImportStats:
    images_seen: int = 0
    images_written: int = 0
    images_empty: int = 0
    labels_seen: int = 0
    labels_written: int = 0
    labels_skipped_category: int = 0
    labels_skipped_geometry: int = 0
    missing_images: int = 0
    object_counts: Counter[str] | None = None

    def __post_init__(self) -> None:
        if self.object_counts is None:
            self.object_counts = Counter()

    def as_dict(self) -> dict[str, Any]:
        return {
            "images_seen": self.images_seen,
            "images_written": self.images_written,
            "images_empty": self.images_empty,
            "labels_seen": self.labels_seen,
            "labels_written": self.labels_written,
            "labels_skipped_category": self.labels_skipped_category,
            "labels_skipped_geometry": self.labels_skipped_geometry,
            "missing_images": self.missing_images,
            "object_counts": dict(self.object_counts or {}),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bdd-root", type=Path, required=True, help="BDD100K root directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output BSD YOLO dataset directory.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    parser.add_argument(
        "--label-template",
        default="labels/det_20/det_{split}.json",
        help="Label JSON path relative to --bdd-root.",
    )
    parser.add_argument(
        "--image-template",
        default="images/100k/{split}",
        help="Image directory path relative to --bdd-root.",
    )
    parser.add_argument("--link-mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    parser.add_argument("--include-empty", action="store_true", help="Keep images with no BSD target labels.")
    parser.add_argument(
        "--empty-ratio",
        type=float,
        default=0.08,
        help="Maximum empty-label images as a fraction of labeled images when --include-empty is set.",
    )
    parser.add_argument("--min-box-px", type=float, default=2.0, help="Minimum width/height in pixels.")
    parser.add_argument("--max-box-area", type=float, default=0.95, help="Maximum normalized box area.")
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected a list in {path}")
    return data


def find_image(image_dir: Path, name: str) -> Path | None:
    direct = image_dir / name
    if direct.exists():
        return direct
    stem = Path(name).stem
    for suffix in IMAGE_SUFFIXES:
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        os.symlink(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def yolo_box_from_bdd(label: dict[str, Any], min_box_px: float, max_box_area: float) -> tuple[float, float, float, float] | None:
    box = label.get("box2d")
    if not isinstance(box, dict):
        return None
    try:
        x1 = float(box["x1"])
        y1 = float(box["y1"])
        x2 = float(box["x2"])
        y2 = float(box["y2"])
    except (KeyError, TypeError, ValueError):
        return None
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    if width < min_box_px or height < min_box_px:
        return None

    # BDD100K 100k images are normally 1280x720. The detection labels do not
    # repeat width/height per image, so keep this importer tied to that official
    # image set unless a later source explicitly supplies dimensions.
    image_w = 1280.0
    image_h = 720.0
    x1 = min(max(x1, 0.0), image_w)
    x2 = min(max(x2, 0.0), image_w)
    y1 = min(max(y1, 0.0), image_h)
    y2 = min(max(y2, 0.0), image_h)
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area = (width * height) / (image_w * image_h)
    if width < min_box_px or height < min_box_px or area > max_box_area:
        return None
    cx = (x1 + width / 2.0) / image_w
    cy = (y1 + height / 2.0) / image_h
    return cx, cy, width / image_w, height / image_h


def write_yaml(output: Path) -> None:
    names = ", ".join(BSD_NAMES)
    (output / "dataset.yaml").write_text(
        "\n".join(
            [
                f"path: {output.as_posix()}",
                "train: train/images",
                "val: val/images",
                "nc: 4",
                f"names: [{names}]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def import_split(args: argparse.Namespace, split: str, rng: random.Random) -> ImportStats:
    label_path = args.bdd_root / args.label_template.format(split=split)
    image_dir = args.bdd_root / args.image_template.format(split=split)
    if not label_path.exists():
        raise FileNotFoundError(label_path)
    if not image_dir.exists():
        raise FileNotFoundError(image_dir)

    records = load_json(label_path)
    stats = ImportStats(images_seen=len(records))
    out_img_dir = args.output / split / "images"
    out_lbl_dir = args.output / split / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    labeled: list[tuple[dict[str, Any], Path, list[str], Counter[str]]] = []
    empty: list[tuple[dict[str, Any], Path]] = []

    for record in records:
        image_name = str(record.get("name", ""))
        image_path = find_image(image_dir, image_name)
        if image_path is None:
            stats.missing_images += 1
            continue

        label_lines: list[str] = []
        image_counts: Counter[str] = Counter()
        for label in record.get("labels", []) or []:
            stats.labels_seen += 1
            category = str(label.get("category", "")).lower()
            if category not in BDD_TO_BSD:
                stats.labels_skipped_category += 1
                continue
            box = yolo_box_from_bdd(label, args.min_box_px, args.max_box_area)
            if box is None:
                stats.labels_skipped_geometry += 1
                continue
            cls_id = BDD_TO_BSD[category]
            cls_name = BSD_NAMES[cls_id]
            cx, cy, width, height = box
            label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}")
            image_counts[cls_name] += 1

        if label_lines:
            labeled.append((record, image_path, label_lines, image_counts))
        else:
            empty.append((record, image_path))

    rng.shuffle(empty)
    if args.include_empty and labeled:
        max_empty = int(len(labeled) * max(0.0, args.empty_ratio))
        kept_empty = empty[:max_empty]
    else:
        kept_empty = []

    for record, image_path, label_lines, image_counts in labeled:
        stem = Path(str(record["name"])).stem
        dst_img = out_img_dir / f"{stem}{image_path.suffix.lower()}"
        dst_lbl = out_lbl_dir / f"{stem}.txt"
        link_or_copy(image_path, dst_img, args.link_mode)
        dst_lbl.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        stats.images_written += 1
        stats.labels_written += len(label_lines)
        stats.object_counts.update(image_counts)

    for record, image_path in kept_empty:
        stem = Path(str(record["name"])).stem
        dst_img = out_img_dir / f"{stem}{image_path.suffix.lower()}"
        dst_lbl = out_lbl_dir / f"{stem}.txt"
        link_or_copy(image_path, dst_img, args.link_mode)
        dst_lbl.write_text("", encoding="utf-8")
        stats.images_written += 1
        stats.images_empty += 1

    return stats


def main() -> int:
    args = parse_args()
    if args.output.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists, pass --overwrite to replace: {args.output}")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    split_stats = {}
    for split in args.splits:
        stats = import_split(args, split, rng)
        split_stats[split] = stats.as_dict()
        print(f"{split}: {json.dumps(split_stats[split], ensure_ascii=False)}")

    if set(args.splits) == {"train", "val"}:
        write_yaml(args.output)

    manifest = {
        "dataset": "bdd100k_det_20_to_bsd",
        "bdd_root": str(args.bdd_root),
        "output": str(args.output),
        "splits": args.splits,
        "class_names": BSD_NAMES,
        "class_mapping": BDD_TO_BSD,
        "filters": {
            "include_empty": args.include_empty,
            "empty_ratio": args.empty_ratio,
            "min_box_px": args.min_box_px,
            "max_box_area": args.max_box_area,
        },
        "link_mode": args.link_mode,
        "seed": args.seed,
        "split_stats": split_stats,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"manifest: {args.output / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
