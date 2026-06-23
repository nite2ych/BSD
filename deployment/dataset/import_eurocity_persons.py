"""
Import EuroCity Persons-style JSON annotations into BSD YOLO format.

The preferred input is the standardized JSON format produced by tools such as
HUMAINT's annotation standardizer:

  {
    "im_name": "<image-name>",
    "city_name": "<city-name>",
    "im_width": 1920,
    "im_height": 1024,
    "agents": [
      {
        "identity": "pedestrian",
        "x0": 100,
        "y0": 200,
        "x1": 140,
        "y1": 320
      }
    ]
  }

EuroCity Persons is used here as a person/rider recall supplement. By default
all selected identities are mapped to BSD class 0 (`person`), and the dataset is
not used as a validation source.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


BSD_NAMES = ["person", "bicycle", "motorcycle", "vehicle"]
BSD_PERSON_ID = 0
IMAGE_SUFFIXES = [".png", ".jpg", ".jpeg"]

DEFAULT_PERSON_IDENTITIES = {
    "pedestrian",
    "person",
    "rider",
    "cyclist",
    "bicyclist",
    "motorcyclist",
    "scooterist",
}

DEFAULT_IGNORE_IDENTITIES = {
    "group",
    "crowd",
    "ignore",
    "ignored",
    "unknown",
    "static_person",
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
    source_identity_counts: Counter[str] | None = None

    def __post_init__(self) -> None:
        if self.object_counts is None:
            self.object_counts = Counter()
        if self.source_identity_counts is None:
            self.source_identity_counts = Counter()

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
            "source_identity_counts": dict(self.source_identity_counts or {}),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ecp-root",
        type=Path,
        required=True,
        help="EuroCity Persons root, used only for manifest traceability.",
    )
    parser.add_argument(
        "--json-root",
        type=Path,
        required=True,
        help="Directory containing standardized JSON annotation files.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help="Image root, e.g. ECP/day/img or a split directory containing images.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output BSD YOLO dataset directory.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val", "test"])
    parser.add_argument("--link-mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    parser.add_argument("--include-empty", action="store_true")
    parser.add_argument("--min-box-px", type=float, default=4.0)
    parser.add_argument("--max-box-area", type=float, default=0.95)
    parser.add_argument(
        "--identity",
        action="append",
        default=[],
        help="Additional source identity to map to BSD person. Can be repeated.",
    )
    parser.add_argument(
        "--ignore-identity",
        action="append",
        default=[],
        help="Additional source identity to ignore. Can be repeated.",
    )
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


def normalize_identity(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def selected_identities(args: argparse.Namespace) -> set[str]:
    return DEFAULT_PERSON_IDENTITIES | {normalize_identity(item) for item in args.identity}


def ignored_identities(args: argparse.Namespace) -> set[str]:
    return DEFAULT_IGNORE_IDENTITIES | {normalize_identity(item) for item in args.ignore_identity}


def iter_json_files(json_root: Path, split: str) -> list[Path]:
    split_dir = json_root / split
    if split_dir.exists():
        return sorted(split_dir.rglob("*.json"))
    return sorted(path for path in json_root.rglob("*.json") if split in path.parts)


def build_image_index(image_root: Path, split: str) -> dict[str, Path]:
    roots = []
    split_dir = image_root / split
    if split_dir.exists():
        roots.append(split_dir)
    roots.append(image_root)

    index: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for image in root.rglob("*"):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            index.setdefault(image.name, image)
            index.setdefault(image.stem, image)
    return index


def find_image(record: dict[str, Any], image_index: dict[str, Path]) -> Path | None:
    im_name = str(record.get("im_name") or record.get("image_name") or record.get("filename") or "")
    if im_name:
        if im_name in image_index:
            return image_index[im_name]
        stem = Path(im_name).stem
        if stem in image_index:
            return image_index[stem]
    key_frame_name = str(record.get("key_frame_name") or "")
    if key_frame_name:
        if key_frame_name in image_index:
            return image_index[key_frame_name]
        stem = Path(key_frame_name).stem
        if stem in image_index:
            return image_index[stem]
    return None


def image_size(record: dict[str, Any], image_path: Path) -> tuple[int, int]:
    try:
        width = int(record.get("im_width") or record.get("width") or 0)
        height = int(record.get("im_height") or record.get("height") or 0)
    except (TypeError, ValueError):
        width, height = 0, 0
    if width > 0 and height > 0:
        return width, height
    with Image.open(image_path) as image:
        return image.size


def agent_box(agent: dict[str, Any], image_w: int, image_h: int, args: argparse.Namespace) -> tuple[str | None, str | None]:
    identity = normalize_identity(agent.get("identity") or agent.get("label") or agent.get("category"))
    if identity in ignored_identities(args):
        return None, "category"
    if identity not in selected_identities(args):
        return None, "category"

    try:
        if {"x0", "y0", "x1", "y1"}.issubset(agent):
            x1 = float(agent["x0"])
            y1 = float(agent["y0"])
            x2 = float(agent["x1"])
            y2 = float(agent["y1"])
        elif {"bbox"}.issubset(agent):
            bbox = agent["bbox"]
            x1 = float(bbox[0])
            y1 = float(bbox[1])
            x2 = x1 + float(bbox[2])
            y2 = y1 + float(bbox[3])
        elif {"box2d"}.issubset(agent):
            box = agent["box2d"]
            x1 = float(box["x1"])
            y1 = float(box["y1"])
            x2 = float(box["x2"])
            y2 = float(box["y2"])
        else:
            return None, "geometry"
    except (TypeError, ValueError, KeyError, IndexError):
        return None, "geometry"

    x1 = min(max(x1, 0.0), float(image_w))
    y1 = min(max(y1, 0.0), float(image_h))
    x2 = min(max(x2, 0.0), float(image_w))
    y2 = min(max(y2, 0.0), float(image_h))
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    area = (bw * bh) / float(image_w * image_h)
    if bw < args.min_box_px or bh < args.min_box_px or area > args.max_box_area:
        return None, "geometry"

    cx = (x1 + bw / 2.0) / float(image_w)
    cy = (y1 + bh / 2.0) / float(image_h)
    return f"{BSD_PERSON_ID} {cx:.6f} {cy:.6f} {bw / image_w:.6f} {bh / image_h:.6f}", None


def agents_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    agents = record.get("agents")
    if isinstance(agents, list):
        return [item for item in agents if isinstance(item, dict)]
    labels = record.get("labels")
    if isinstance(labels, list):
        return [item for item in labels if isinstance(item, dict)]
    annotations = record.get("annotations")
    if isinstance(annotations, list):
        return [item for item in annotations if isinstance(item, dict)]
    return []


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


def import_split(args: argparse.Namespace, split: str) -> ImportStats:
    stats = ImportStats()
    out_img_dir = args.output / split / "images"
    out_lbl_dir = args.output / split / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    image_index = build_image_index(args.image_root, split)

    for json_path in iter_json_files(args.json_root, split):
        stats.images_seen += 1
        record = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(record, dict):
            continue
        image_path = find_image(record, image_index)
        if image_path is None:
            stats.missing_images += 1
            continue
        image_w, image_h = image_size(record, image_path)
        agents = agents_from_record(record)
        label_lines: list[str] = []
        for agent in agents:
            stats.labels_seen += 1
            identity = normalize_identity(agent.get("identity") or agent.get("label") or agent.get("category"))
            stats.source_identity_counts[identity or "unknown"] += 1
            label_line, skip_reason = agent_box(agent, image_w, image_h, args)
            if skip_reason == "category":
                stats.labels_skipped_category += 1
                continue
            if skip_reason == "geometry":
                stats.labels_skipped_geometry += 1
                continue
            if label_line is not None:
                label_lines.append(label_line)
                stats.labels_written += 1
                stats.object_counts["person"] += 1

        if not label_lines and not args.include_empty:
            continue
        if not label_lines:
            stats.images_empty += 1

        stem = image_path.stem
        dst_img = out_img_dir / f"{stem}{image_path.suffix.lower()}"
        dst_lbl = out_lbl_dir / f"{stem}.txt"
        link_or_copy(image_path, dst_img, args.link_mode)
        dst_lbl.write_text(("\n".join(label_lines) + "\n") if label_lines else "", encoding="utf-8")
        stats.images_written += 1
    return stats


def main() -> int:
    args = parse_args()
    if args.output.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists, pass --overwrite to replace: {args.output}")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset": "eurocity_persons_to_bsd",
        "source": {
            "ecp_root": str(args.ecp_root),
            "json_root": str(args.json_root),
            "image_root": str(args.image_root),
        },
        "output": str(args.output),
        "filters": {
            "splits": args.splits,
            "selected_identities": sorted(selected_identities(args)),
            "ignored_identities": sorted(ignored_identities(args)),
            "include_empty": args.include_empty,
            "min_box_px": args.min_box_px,
            "max_box_area": args.max_box_area,
        },
        "splits": {},
    }
    for split in args.splits:
        manifest["splits"][split] = import_split(args, split).as_dict()
    write_yaml(args.output)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
