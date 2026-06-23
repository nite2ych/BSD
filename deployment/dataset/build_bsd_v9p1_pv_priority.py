"""
Build a person/vehicle-priority dataset for BSD v9.1 fine-tuning.

This script keeps the existing BSD YOLO directory format and starts from the
current v6 public/KITTI dataset. It can add reviewed extra sets later, but it
already records the person/vehicle sampling policy and cleans duplicate labels
from the proxy validation split.
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


CLASS_NAMES = ["person", "bicycle", "motorcycle", "vehicle"]
PV_CLASSES = {0, 3}
RARE_CLASSES = {1, 2}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class ImageSet:
    name: str
    image_dir: Path
    label_dir: Path


@dataclass
class Label:
    cls: int
    x: float
    y: float
    w: float
    h: float

    @property
    def key(self) -> tuple[int, int, int, int, int]:
        return (
            self.cls,
            round(self.x * 1_000_000),
            round(self.y * 1_000_000),
            round(self.w * 1_000_000),
            round(self.h * 1_000_000),
        )

    @property
    def area(self) -> float:
        return self.w * self.h

    def line(self) -> str:
        return f"{self.cls} {self.x:.6f} {self.y:.6f} {self.w:.6f} {self.h:.6f}\n"


@dataclass
class Sample:
    image: Path
    label: Path
    source: str
    labels: list[Label]

    @property
    def classes(self) -> set[int]:
        return {label.cls for label in self.labels}


def parse_extra_set(raw: str) -> ImageSet:
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("extra set must be name:image_dir:label_dir")
    return ImageSet(parts[0], Path(parts[1]), Path(parts[2]))


def parse_labels(path: Path, *, dedupe: bool) -> tuple[list[Label], int]:
    labels: list[Label] = []
    duplicate_count = 0
    seen = set()
    if not path.exists():
        return labels, duplicate_count
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) < 5:
            continue
        try:
            label = Label(
                cls=int(float(parts[0])),
                x=float(parts[1]),
                y=float(parts[2]),
                w=float(parts[3]),
                h=float(parts[4]),
            )
        except ValueError:
            continue
        if label.cls < 0 or label.cls >= len(CLASS_NAMES):
            continue
        if not (0.0 <= label.x <= 1.0 and 0.0 <= label.y <= 1.0 and 0.0 < label.w <= 1.0 and 0.0 < label.h <= 1.0):
            continue
        if dedupe and label.key in seen:
            duplicate_count += 1
            continue
        seen.add(label.key)
        labels.append(label)
    return labels, duplicate_count


def collect(image_dir: Path, label_dir: Path, source: str, *, dedupe: bool) -> tuple[list[Sample], dict[str, int]]:
    samples: list[Sample] = []
    stats = {"missing_labels": 0, "duplicate_labels_removed": 0, "empty_labels": 0}
    for image in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        label = label_dir / f"{image.stem}.txt"
        if not label.exists():
            stats["missing_labels"] += 1
            continue
        labels, duplicates = parse_labels(label, dedupe=dedupe)
        stats["duplicate_labels_removed"] += duplicates
        if not labels:
            stats["empty_labels"] += 1
        samples.append(Sample(image=image, label=label, source=source, labels=labels))
    return samples, stats


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


def write_sample(sample: Sample, out_split: Path, mode: str, prefix: str = "") -> None:
    stem = f"{prefix}{sample.image.stem}" if prefix else sample.image.stem
    image_dst = out_split / "images" / f"{stem}{sample.image.suffix.lower()}"
    label_dst = out_split / "labels" / f"{stem}.txt"
    link_or_copy(sample.image, image_dst, mode)
    label_dst.parent.mkdir(parents=True, exist_ok=True)
    label_dst.write_text("".join(label.line() for label in sample.labels), encoding="utf-8")


def summarize(samples: list[Sample]) -> dict[str, object]:
    image_class_counts = Counter()
    object_counts = Counter()
    area_buckets = Counter()
    source_counts = Counter(sample.source for sample in samples)
    empty = 0
    for sample in samples:
        if not sample.labels:
            empty += 1
        for cls in sample.classes:
            image_class_counts[CLASS_NAMES[cls]] += 1
        for label in sample.labels:
            object_counts[CLASS_NAMES[label.cls]] += 1
            if label.area < 0.005:
                area_buckets["tiny"] += 1
            elif label.area < 0.02:
                area_buckets["small"] += 1
            elif label.area < 0.15:
                area_buckets["medium"] += 1
            else:
                area_buckets["large"] += 1
    return {
        "images": len(samples),
        "empty_labels": empty,
        "sources": dict(source_counts),
        "image_class_counts": dict(image_class_counts),
        "object_counts": dict(object_counts),
        "area_buckets": dict(area_buckets),
    }


def pick_extra(samples: list[Sample], args: argparse.Namespace) -> list[Sample]:
    rng = random.Random(args.seed)
    rare = [sample for sample in samples if sample.classes & RARE_CLASSES]
    pv_mixed = [sample for sample in samples if sample.classes & PV_CLASSES and sample.classes - PV_CLASSES]
    person_vehicle = [sample for sample in samples if {0, 3}.issubset(sample.classes)]
    person_only = [sample for sample in samples if sample.classes == {0}]
    vehicle_only = [sample for sample in samples if sample.classes == {3}]
    empty = [sample for sample in samples if not sample.labels]

    selected: list[Sample] = []
    selected.extend(rare)
    selected.extend(pv_mixed)
    for group, limit in [
        (person_vehicle, args.max_extra_person_vehicle),
        (person_only, args.max_extra_person_only),
        (vehicle_only, args.max_extra_vehicle_only),
        (empty, args.max_extra_empty),
    ]:
        rng.shuffle(group)
        selected.extend(group[:limit])

    by_key = {}
    for sample in selected:
        by_key[(sample.image.resolve(), sample.source)] = sample
    return sorted(by_key.values(), key=lambda item: (item.source, item.image.name))


def yaml_path(root: Path, value: Path) -> str:
    try:
        return value.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return value.as_posix()


def write_dataset_yaml(path: Path, train_images: Path, val_images: Path) -> None:
    root = path.parent
    names = ", ".join(CLASS_NAMES)
    path.write_text(
        "\n".join(
            [
                f"path: {root.as_posix()}",
                f"train: {yaml_path(root, train_images)}",
                f"val: {yaml_path(root, val_images)}",
                "nc: 4",
                f"names: [{names}]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("/root/autodl-tmp/BSD/datasets/bsd_v6_public_kitti"))
    parser.add_argument("--coco-val", type=Path, default=Path("/root/autodl-tmp/BSD/datasets/bsd/val/images"))
    parser.add_argument("--output", type=Path, default=Path("/root/autodl-tmp/BSD/datasets/bsd_v9p1_pv_priority"))
    parser.add_argument("--extra-set", action="append", type=parse_extra_set, default=[])
    parser.add_argument("--link-mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--max-extra-person-vehicle", type=int, default=5000)
    parser.add_argument("--max-extra-person-only", type=int, default=3000)
    parser.add_argument("--max-extra-vehicle-only", type=int, default=1200)
    parser.add_argument("--max-extra-empty", type=int, default=1200)
    args = parser.parse_args()

    if args.output.exists():
        shutil.rmtree(args.output)
    for split in ["train", "bdd_proxy_val"]:
        (args.output / split / "images").mkdir(parents=True, exist_ok=True)
        (args.output / split / "labels").mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "base": str(args.base),
        "output": str(args.output),
        "seed": args.seed,
        "policy": {
            "primary_classes": ["person", "vehicle"],
            "rare_classes_kept_as_auxiliary": ["bicycle", "motorcycle"],
            "max_extra_person_vehicle": args.max_extra_person_vehicle,
            "max_extra_person_only": args.max_extra_person_only,
            "max_extra_vehicle_only": args.max_extra_vehicle_only,
            "max_extra_empty": args.max_extra_empty,
        },
        "extra_sets": [],
    }

    base_train, base_train_stats = collect(args.base / "train/images", args.base / "train/labels", "base_train", dedupe=True)
    base_proxy, base_proxy_stats = collect(
        args.base / "bdd_proxy_val/images",
        args.base / "bdd_proxy_val/labels",
        "bdd_proxy_val",
        dedupe=True,
    )
    for sample in base_train:
        write_sample(sample, args.output / "train", args.link_mode)
    for sample in base_proxy:
        write_sample(sample, args.output / "bdd_proxy_val", args.link_mode)

    train_samples = list(base_train)
    for extra in args.extra_set:
        samples, stats = collect(extra.image_dir, extra.label_dir, extra.name, dedupe=True)
        selected = pick_extra(samples, args)
        for sample in selected:
            write_sample(sample, args.output / "train", args.link_mode, prefix=f"{extra.name}_")
        train_samples.extend(selected)
        manifest["extra_sets"].append(
            {
                "name": extra.name,
                "image_dir": str(extra.image_dir),
                "label_dir": str(extra.label_dir),
                "input_summary": summarize(samples),
                "selected_summary": summarize(selected),
                "read_stats": stats,
            }
        )

    write_dataset_yaml(args.output / "dataset.yaml", args.output / "train/images", args.coco_val)
    write_dataset_yaml(args.output / "bdd_proxy_val.yaml", args.output / "train/images", args.output / "bdd_proxy_val/images")

    manifest["base_read_stats"] = {
        "train": base_train_stats,
        "bdd_proxy_val": base_proxy_stats,
    }
    manifest["train_summary"] = summarize(train_samples)
    manifest["bdd_proxy_val_summary"] = summarize(base_proxy)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
