"""
Import KITTI Object Detection into BSD YOLO format.

Expected KITTI layout:
  <kitti-root>/
    training/image_2/*.png
    training/label_2/*.txt

The script maps KITTI labels to BSD classes:
  Pedestrian, Person_sitting -> person
  Cyclist -> bicycle
  Car, Van, Truck, Tram -> vehicle

KITTI does not have a motorcycle class. `DontCare` is ignored.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


BSD_NAMES = ["person", "bicycle", "motorcycle", "vehicle"]
KITTI_TO_BSD = {
    "pedestrian": 0,
    "person_sitting": 0,
    "cyclist": 1,
    "car": 3,
    "van": 3,
    "truck": 3,
    "tram": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kitti-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--link-mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    parser.add_argument("--min-box-px", type=float, default=2.0)
    parser.add_argument("--max-box-area", type=float, default=0.95)
    parser.add_argument("--include-empty", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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


def parse_label_lines(raw_lines: list[str], image_w: int, image_h: int, min_box_px: float, max_box_area: float) -> tuple[list[str], Counter[str], dict[str, int]]:
    lines: list[str] = []
    counts: Counter[str] = Counter()
    skipped = {"category": 0, "geometry": 0}
    for raw in raw_lines:
        parts = raw.strip().split()
        if len(parts) < 8:
            continue
        category = parts[0].lower()
        if category == "dontcare":
            continue
        if category not in KITTI_TO_BSD:
            skipped["category"] += 1
            continue
        try:
            x1, y1, x2, y2 = map(float, parts[4:8])
        except ValueError:
            skipped["geometry"] += 1
            continue
        x1 = min(max(x1, 0.0), float(image_w))
        x2 = min(max(x2, 0.0), float(image_w))
        y1 = min(max(y1, 0.0), float(image_h))
        y2 = min(max(y2, 0.0), float(image_h))
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        area = (bw * bh) / float(image_w * image_h)
        if bw < min_box_px or bh < min_box_px or area > max_box_area:
            skipped["geometry"] += 1
            continue
        cls_id = KITTI_TO_BSD[category]
        cls_name = BSD_NAMES[cls_id]
        cx = (x1 + bw / 2.0) / image_w
        cy = (y1 + bh / 2.0) / image_h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw / image_w:.6f} {bh / image_h:.6f}")
        counts[cls_name] += 1
    return lines, counts, skipped


def parse_label(label_path: Path, image_w: int, image_h: int, min_box_px: float, max_box_area: float) -> tuple[list[str], Counter[str], dict[str, int]]:
    if not label_path.exists():
        return [], Counter(), {"category": 0, "geometry": 0}
    return parse_label_lines(
        label_path.read_text(encoding="utf-8", errors="replace").splitlines(),
        image_w,
        image_h,
        min_box_px,
        max_box_area,
    )


def find_zip_member(members: dict[str, str], stem: str) -> str | None:
    return members.get(stem)


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


def main() -> int:
    args = parse_args()
    image_dir = args.kitti_root / "training" / "image_2"
    label_dir = args.kitti_root / "training" / "label_2"
    image_zip_path = args.kitti_root / "data_object_image_2.zip"
    label_zip_path = args.kitti_root / "data_object_label_2.zip"
    use_zip = not image_dir.exists() or not label_dir.exists()
    if use_zip:
        if not image_zip_path.exists():
            raise FileNotFoundError(image_dir)
        if not label_zip_path.exists():
            raise FileNotFoundError(label_dir)
    if args.output.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists, pass --overwrite to replace: {args.output}")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    image_zip = zipfile.ZipFile(image_zip_path) if use_zip else None
    label_zip = zipfile.ZipFile(label_zip_path) if use_zip else None
    if use_zip:
        image_members = {
            Path(name).stem: name
            for name in image_zip.namelist()
            if name.lower().startswith("training/image_2/") and Path(name).suffix.lower() in {".png", ".jpg", ".jpeg"}
        }
        label_members = {
            Path(name).stem: name
            for name in label_zip.namelist()
            if name.lower().startswith("training/label_2/") and Path(name).suffix.lower() == ".txt"
        }
        items = sorted(image_members)
    else:
        image_members = {}
        label_members = {}
        items = sorted(p.stem for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    rng = random.Random(args.seed)
    rng.shuffle(items)
    val_count = int(len(items) * args.val_ratio)
    val_stems = set(items[:val_count])

    stats: dict[str, Any] = {
        "dataset": "kitti_object_to_bsd",
        "source": str(args.kitti_root),
        "output": str(args.output),
        "filters": {
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "min_box_px": args.min_box_px,
            "max_box_area": args.max_box_area,
            "include_empty": args.include_empty,
        },
        "splits": {},
    }
    total_counts: Counter[str] = Counter()

    for stem in items:
        split = "val" if stem in val_stems else "train"
        out_img_dir = args.output / split / "images"
        out_lbl_dir = args.output / split / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)
        stats["splits"].setdefault(
            split,
            {
                "images_seen": 0,
                "images_written": 0,
                "images_empty": 0,
                "labels_written": 0,
                "object_counts": {},
                "skipped_category": 0,
                "skipped_geometry": 0,
            },
        )
        split_stats = stats["splits"][split]
        split_stats["images_seen"] += 1

        if use_zip:
            image_member = find_zip_member(image_members, stem)
            if image_member is None:
                continue
            image_bytes = image_zip.read(image_member)
            with Image.open(BytesIO(image_bytes)) as im:
                image_w, image_h = im.size
                image_suffix = Path(image_member).suffix.lower()
            label_member = find_zip_member(label_members, stem)
            if label_member is None:
                raw_lines: list[str] = []
            else:
                raw_lines = label_zip.read(label_member).decode("utf-8", errors="replace").splitlines()
            label_lines, counts, skipped = parse_label_lines(raw_lines, image_w, image_h, args.min_box_px, args.max_box_area)
        else:
            image = image_dir / f"{stem}.png"
            if not image.exists():
                for suffix in [".jpg", ".jpeg", ".png"]:
                    candidate = image_dir / f"{stem}{suffix}"
                    if candidate.exists():
                        image = candidate
                        break
            with Image.open(image) as im:
                image_w, image_h = im.size
            image_suffix = image.suffix.lower()
            label_lines, counts, skipped = parse_label(
                label_dir / f"{stem}.txt",
                image_w,
                image_h,
                args.min_box_px,
                args.max_box_area,
            )
        split_stats["skipped_category"] += skipped["category"]
        split_stats["skipped_geometry"] += skipped["geometry"]
        if not label_lines and not args.include_empty:
            continue

        dst_img = out_img_dir / f"{stem}{image_suffix}"
        dst_lbl = out_lbl_dir / f"{stem}.txt"
        if use_zip:
            dst_img.write_bytes(image_bytes)
        else:
            link_or_copy(image, dst_img, args.link_mode)
        dst_lbl.write_text(("\n".join(label_lines) + "\n") if label_lines else "", encoding="utf-8")
        split_stats["images_written"] += 1
        split_stats["labels_written"] += len(label_lines)
        if not label_lines:
            split_stats["images_empty"] += 1
        current = Counter(split_stats["object_counts"])
        current.update(counts)
        split_stats["object_counts"] = dict(current)
        total_counts.update(counts)

    stats["object_counts"] = dict(total_counts)
    write_yaml(args.output)
    (args.output / "manifest.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if image_zip is not None:
        image_zip.close()
    if label_zip is not None:
        label_zip.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
