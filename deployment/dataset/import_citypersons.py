"""
Import CityPersons annotations into BSD YOLO format.

Expected inputs:
  CityPersons annotations:
    <citypersons-root>/gtBboxCityPersons.mat

  Cityscapes leftImg8bit images:
    <cityscapes-leftimg-root>/train/<city>/<image>_leftImg8bit.png
    <cityscapes-leftimg-root>/val/<city>/<image>_leftImg8bit.png

The script maps CityPersons pedestrian-like classes to BSD's `person` class:
  pedestrian -> person
  rider -> person by default

Sitting/other/group classes are ignored by default because their boxes are less
consistent for BSD detection. They can be enabled explicitly for experiments.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


BSD_NAMES = ["person", "bicycle", "motorcycle", "vehicle"]
BSD_PERSON_ID = 0
IMAGE_SUFFIXES = [".png", ".jpg", ".jpeg"]

# CityPersons full annotation columns are commonly:
#   class, x, y, w, h, instance_id, x_vis, y_vis, w_vis, h_vis
CLASS_NAMES = {
    1: "pedestrian",
    2: "rider",
    3: "sitting_person",
    4: "other_person",
    5: "group",
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
    source_class_counts: Counter[str] | None = None

    def __post_init__(self) -> None:
        if self.object_counts is None:
            self.object_counts = Counter()
        if self.source_class_counts is None:
            self.source_class_counts = Counter()

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
            "source_class_counts": dict(self.source_class_counts or {}),
        }


@dataclass(frozen=True)
class AnnotationRecord:
    split: str
    city: str
    image_name: str
    boxes: list[list[float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--citypersons-root",
        type=Path,
        required=True,
        help="Directory containing gtBboxCityPersons.mat, or the .mat file itself.",
    )
    parser.add_argument(
        "--cityscapes-leftimg-root",
        type=Path,
        required=True,
        help="Cityscapes leftImg8bit root, e.g. /data/cityscapes/leftImg8bit.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output BSD YOLO dataset directory.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    parser.add_argument("--link-mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    parser.add_argument("--box-mode", choices=["full", "visible"], default="full")
    parser.add_argument("--exclude-rider", action="store_true", help="Do not map CityPersons rider to BSD person.")
    parser.add_argument("--include-sitting", action="store_true", help="Map CityPersons sitting person to BSD person.")
    parser.add_argument("--include-other-person", action="store_true", help="Map CityPersons other person to BSD person.")
    parser.add_argument("--include-empty", action="store_true", help="Keep images without selected BSD labels.")
    parser.add_argument("--min-box-px", type=float, default=4.0)
    parser.add_argument("--max-box-area", type=float, default=0.95)
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


def to_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    if hasattr(value, "item"):
        try:
            return to_text(value.item())
        except ValueError:
            pass
    if isinstance(value, Iterable):
        items = list(value)
        if len(items) == 1:
            return to_text(items[0])
    return str(value)


def get_field(item: Any, names: list[str]) -> Any:
    for name in names:
        if hasattr(item, name):
            return getattr(item, name)
        if isinstance(item, dict) and name in item:
            return item[name]
        if hasattr(item, "dtype") and getattr(item.dtype, "names", None) and name in item.dtype.names:
            return item[name]
    raise KeyError(f"missing any field from {names}")


def as_records(value: Any) -> list[Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("CityPersons import requires numpy and scipy.") from exc
    return list(np.atleast_1d(value).ravel())


def boxes_to_rows(value: Any) -> list[list[float]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("CityPersons import requires numpy and scipy.") from exc
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return []
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr.tolist()


def annotation_mat_path(citypersons_root: Path) -> Path:
    if citypersons_root.is_file():
        return citypersons_root
    candidates = [
        citypersons_root / "gtBboxCityPersons.mat",
        citypersons_root / "gtBbox_cityPersons_trainval" / "gtBboxCityPersons.mat",
        citypersons_root / "annotations" / "gtBboxCityPersons.mat",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(candidates[0])


def load_annotations(citypersons_root: Path, splits: list[str]) -> list[AnnotationRecord]:
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise SystemExit("CityPersons import requires scipy. Install scipy on the data-prep machine.") from exc

    mat_path = annotation_mat_path(citypersons_root)
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    records: list[AnnotationRecord] = []
    for split in splits:
        key_candidates = [f"anno_{split}_aligned", f"anno_{split}", split]
        key = next((candidate for candidate in key_candidates if candidate in data), None)
        if key is None:
            raise KeyError(f"could not find CityPersons split {split!r} in {mat_path}")
        for item in as_records(data[key]):
            city = to_text(get_field(item, ["city_name", "cityname", "city"]))
            image_name = to_text(get_field(item, ["im_name", "img_name", "image_name", "filename"]))
            boxes = boxes_to_rows(get_field(item, ["bbs", "bb", "boxes"]))
            records.append(AnnotationRecord(split=split, city=city, image_name=image_name, boxes=boxes))
    return records


def selected_classes(args: argparse.Namespace) -> set[int]:
    selected = {1}
    if not args.exclude_rider:
        selected.add(2)
    if args.include_sitting:
        selected.add(3)
    if args.include_other_person:
        selected.add(4)
    return selected


def box_from_row(
    row: list[float],
    *,
    image_w: int,
    image_h: int,
    args: argparse.Namespace,
) -> tuple[str | None, str | None]:
    if len(row) < 5:
        return None, "geometry"
    source_cls = int(row[0])
    if source_cls not in selected_classes(args):
        return None, "category"

    if args.box_mode == "visible" and len(row) >= 10:
        x, y, bw, bh = row[6], row[7], row[8], row[9]
    else:
        x, y, bw, bh = row[1], row[2], row[3], row[4]

    x1 = min(max(float(x), 0.0), float(image_w))
    y1 = min(max(float(y), 0.0), float(image_h))
    x2 = min(max(float(x) + float(bw), 0.0), float(image_w))
    y2 = min(max(float(y) + float(bh), 0.0), float(image_h))
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    area = (bw * bh) / float(image_w * image_h)
    if bw < args.min_box_px or bh < args.min_box_px or area > args.max_box_area:
        return None, "geometry"
    cx = (x1 + bw / 2.0) / float(image_w)
    cy = (y1 + bh / 2.0) / float(image_h)
    return f"{BSD_PERSON_ID} {cx:.6f} {cy:.6f} {bw / image_w:.6f} {bh / image_h:.6f}", None


def find_image(root: Path, record: AnnotationRecord) -> Path | None:
    direct = root / record.split / record.city / record.image_name
    if direct.exists():
        return direct
    stem = Path(record.image_name).stem
    city_dir = root / record.split / record.city
    for suffix in IMAGE_SUFFIXES:
        candidate = city_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    matches = list((root / record.split).glob(f"*/{record.image_name}"))
    if matches:
        return matches[0]
    return None


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
    if args.output.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists, pass --overwrite to replace: {args.output}")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    records = load_annotations(args.citypersons_root, args.splits)
    stats = {
        "dataset": "citypersons_to_bsd",
        "source": {
            "citypersons_root": str(args.citypersons_root),
            "cityscapes_leftimg_root": str(args.cityscapes_leftimg_root),
        },
        "output": str(args.output),
        "filters": {
            "splits": args.splits,
            "box_mode": args.box_mode,
            "selected_classes": sorted(CLASS_NAMES[cls] for cls in selected_classes(args)),
            "include_empty": args.include_empty,
            "min_box_px": args.min_box_px,
            "max_box_area": args.max_box_area,
        },
        "splits": {},
    }

    for split in args.splits:
        (args.output / split / "images").mkdir(parents=True, exist_ok=True)
        (args.output / split / "labels").mkdir(parents=True, exist_ok=True)
        split_stats = ImportStats()
        for record in [item for item in records if item.split == split]:
            split_stats.images_seen += 1
            image_path = find_image(args.cityscapes_leftimg_root, record)
            if image_path is None:
                split_stats.missing_images += 1
                continue
            with Image.open(image_path) as image:
                image_w, image_h = image.size

            label_lines: list[str] = []
            for row in record.boxes:
                split_stats.labels_seen += 1
                source_name = CLASS_NAMES.get(int(row[0]) if row else -1, "unknown")
                split_stats.source_class_counts[source_name] += 1
                label_line, skip_reason = box_from_row(row, image_w=image_w, image_h=image_h, args=args)
                if skip_reason == "category":
                    split_stats.labels_skipped_category += 1
                    continue
                if skip_reason == "geometry":
                    split_stats.labels_skipped_geometry += 1
                    continue
                if label_line is not None:
                    label_lines.append(label_line)
                    split_stats.labels_written += 1
                    split_stats.object_counts["person"] += 1

            if not label_lines and not args.include_empty:
                continue
            if not label_lines:
                split_stats.images_empty += 1

            stem = image_path.stem
            dst_img = args.output / split / "images" / f"{stem}{image_path.suffix.lower()}"
            dst_lbl = args.output / split / "labels" / f"{stem}.txt"
            link_or_copy(image_path, dst_img, args.link_mode)
            dst_lbl.write_text(("\n".join(label_lines) + "\n") if label_lines else "", encoding="utf-8")
            split_stats.images_written += 1

        stats["splits"][split] = split_stats.as_dict()

    write_yaml(args.output)
    (args.output / "manifest.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
