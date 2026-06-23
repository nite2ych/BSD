"""
Build the BSD v5 balanced dataset from the existing bsd_merged dataset.

Intended to run on the training server. The script keeps the current COCO/BDD
sources reproducible, filters risky labels, creates a BDD proxy validation split,
and writes manifests for later audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


CLASS_NAMES = ["person", "bicycle", "motorcycle", "vehicle"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class Label:
    cls: int
    x: float
    y: float
    w: float
    h: float
    conf: float | None = None

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
    def stem(self) -> str:
        return self.image.stem

    @property
    def classes(self) -> set[int]:
        return {lb.cls for lb in self.labels}


def parse_label_file(
    path: Path,
    *,
    max_box_area: float,
    max_box_side: float,
    min_box_side: float,
    min_pseudo_conf: float,
    source: str,
) -> tuple[list[Label], list[dict[str, str]]]:
    labels: list[Label] = []
    rejects: list[dict[str, str]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) < 5:
            rejects.append({"label": str(path), "line": str(lineno), "reason": "short_line", "raw": raw})
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = (float(v) for v in parts[1:5])
            conf = float(parts[5]) if len(parts) >= 6 else None
        except ValueError:
            rejects.append({"label": str(path), "line": str(lineno), "reason": "parse_error", "raw": raw})
            continue

        reason = None
        if cls < 0 or cls >= len(CLASS_NAMES):
            reason = "bad_class"
        elif not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            reason = "bad_box_range"
        elif w < min_box_side or h < min_box_side:
            reason = "too_small"
        elif w > max_box_side or h > max_box_side or (w * h) > max_box_area:
            reason = "too_large"
        elif source == "bdd" and conf is not None and conf < min_pseudo_conf:
            reason = "low_pseudo_conf"

        if reason:
            rejects.append({"label": str(path), "line": str(lineno), "reason": reason, "raw": raw})
            continue
        labels.append(Label(cls=cls, x=x, y=y, w=w, h=h, conf=conf))
    return labels, rejects


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


def write_sample(sample: Sample, out_split: Path, mode: str) -> None:
    image_dst = out_split / "images" / sample.image.name
    label_dst = out_split / "labels" / f"{sample.stem}.txt"
    link_or_copy(sample.image, image_dst, mode)
    label_dst.parent.mkdir(parents=True, exist_ok=True)
    label_dst.write_text("".join(lb.line() for lb in sample.labels), encoding="utf-8")


def summarize(samples: list[Sample]) -> dict[str, object]:
    source_counts = Counter(s.source for s in samples)
    image_class_counts = Counter()
    object_counts = Counter()
    area_buckets = Counter()
    for sample in samples:
        for cls in sample.classes:
            image_class_counts[CLASS_NAMES[cls]] += 1
        for lb in sample.labels:
            object_counts[CLASS_NAMES[lb.cls]] += 1
            if lb.area < 0.005:
                area_buckets["tiny"] += 1
            elif lb.area < 0.02:
                area_buckets["small"] += 1
            elif lb.area < 0.15:
                area_buckets["medium"] += 1
            else:
                area_buckets["large"] += 1
    return {
        "images": len(samples),
        "sources": dict(source_counts),
        "image_class_counts": dict(image_class_counts),
        "object_counts": dict(object_counts),
        "area_buckets": dict(area_buckets),
    }


def collect_samples(args: argparse.Namespace) -> tuple[list[Sample], list[dict[str, str]]]:
    image_dir = args.input / "images"
    label_dir = args.input / "labels"
    samples: list[Sample] = []
    rejects: list[dict[str, str]] = []
    for image in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        source = "bdd" if image.name.startswith("bdd_") else "coco" if image.name.startswith("coco_") else "other"
        label = label_dir / f"{image.stem}.txt"
        if not label.exists():
            rejects.append({"label": str(label), "line": "0", "reason": "missing_label", "raw": image.name})
            continue
        labels, label_rejects = parse_label_file(
            label,
            max_box_area=args.max_box_area,
            max_box_side=args.max_box_side,
            min_box_side=args.min_box_side,
            min_pseudo_conf=args.min_pseudo_conf,
            source=source,
        )
        rejects.extend(label_rejects)
        if not labels:
            rejects.append({"label": str(label), "line": "0", "reason": "empty_after_filter", "raw": image.name})
            continue
        samples.append(Sample(image=image, label=label, source=source, labels=labels))
    return samples, rejects


def select_train_and_proxy(samples: list[Sample], args: argparse.Namespace) -> tuple[list[Sample], list[Sample]]:
    rng = random.Random(args.seed)
    bdd_samples = [s for s in samples if s.source == "bdd"]
    rng.shuffle(bdd_samples)
    proxy = bdd_samples[: min(args.bdd_proxy_count, len(bdd_samples))]
    proxy_stems = {s.stem for s in proxy}
    pool = [s for s in samples if s.stem not in proxy_stems]

    rare = [s for s in pool if s.classes & {1, 2}]
    rare_stems = {s.stem for s in rare}
    mixed_person = [s for s in pool if s.stem not in rare_stems and 0 in s.classes and len(s.classes) > 1]
    mixed_person_stems = {s.stem for s in mixed_person}
    person_only = [s for s in pool if s.stem not in rare_stems and s.classes == {0}]
    person_only_stems = {s.stem for s in person_only}
    vehicle_only = [s for s in pool if s.stem not in rare_stems and s.classes == {3}]
    vehicle_only_stems = {s.stem for s in vehicle_only}
    other = [
        s
        for s in pool
        if s.stem not in rare_stems
        and s.stem not in mixed_person_stems
        and s.stem not in person_only_stems
        and s.stem not in vehicle_only_stems
    ]

    for group in (mixed_person, person_only, vehicle_only, other):
        rng.shuffle(group)

    train = []
    train.extend(rare)
    train.extend(mixed_person[: args.max_mixed_person])
    train.extend(person_only[: args.max_person_only])
    train.extend(vehicle_only[: args.max_vehicle_only])
    train.extend(other[: args.max_other])

    # Stable de-dup and ordering after randomized selection.
    dedup = {s.stem: s for s in train}
    return sorted(dedup.values(), key=lambda s: s.image.name), sorted(proxy, key=lambda s: s.image.name)


def yaml_path(dataset_root: Path, value: Path) -> str:
    try:
        return value.resolve().relative_to(dataset_root.resolve()).as_posix()
    except ValueError:
        return value.as_posix()


def write_dataset_yaml(path: Path, train_images: Path, val_images: Path) -> None:
    names = ", ".join(CLASS_NAMES)
    dataset_root = path.parent
    path.write_text(
        "\n".join(
            [
                f"path: {dataset_root.as_posix()}",
                f"train: {yaml_path(dataset_root, train_images)}",
                f"val: {yaml_path(dataset_root, val_images)}",
                "nc: 4",
                f"names: [{names}]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("/root/autodl-tmp/BSD/datasets/bsd_merged"))
    parser.add_argument("--coco-val", type=Path, default=Path("/root/autodl-tmp/BSD/datasets/bsd/val/images"))
    parser.add_argument("--output", type=Path, default=Path("/root/autodl-tmp/BSD/datasets/bsd_v5_balanced"))
    parser.add_argument("--link-mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bdd-proxy-count", type=int, default=600)
    parser.add_argument("--max-mixed-person", type=int, default=5000)
    parser.add_argument("--max-person-only", type=int, default=3500)
    parser.add_argument("--max-vehicle-only", type=int, default=2500)
    parser.add_argument("--max-other", type=int, default=2000)
    parser.add_argument("--max-box-area", type=float, default=0.85)
    parser.add_argument("--max-box-side", type=float, default=0.98)
    parser.add_argument("--min-box-side", type=float, default=0.002)
    parser.add_argument("--min-pseudo-conf", type=float, default=0.25)
    args = parser.parse_args()

    samples, rejects = collect_samples(args)
    train, proxy = select_train_and_proxy(samples, args)

    if args.output.exists():
        shutil.rmtree(args.output)
    for split in ["train", "bdd_proxy_val"]:
        (args.output / split / "images").mkdir(parents=True, exist_ok=True)
        (args.output / split / "labels").mkdir(parents=True, exist_ok=True)

    for sample in train:
        write_sample(sample, args.output / "train", args.link_mode)
    for sample in proxy:
        write_sample(sample, args.output / "bdd_proxy_val", args.link_mode)

    write_dataset_yaml(args.output / "dataset.yaml", args.output / "train/images", args.coco_val)
    write_dataset_yaml(args.output / "bdd_proxy_val.yaml", args.output / "train/images", args.output / "bdd_proxy_val/images")

    manifest = {
        "input": str(args.input),
        "output": str(args.output),
        "seed": args.seed,
        "filters": {
            "max_box_area": args.max_box_area,
            "max_box_side": args.max_box_side,
            "min_box_side": args.min_box_side,
            "min_pseudo_conf": args.min_pseudo_conf,
        },
        "selection": {
            "bdd_proxy_count": args.bdd_proxy_count,
            "max_mixed_person": args.max_mixed_person,
            "max_person_only": args.max_person_only,
            "max_vehicle_only": args.max_vehicle_only,
            "max_other": args.max_other,
        },
        "source_summary_after_clean": summarize(samples),
        "train_summary": summarize(train),
        "bdd_proxy_val_summary": summarize(proxy),
        "reject_count": len(rejects),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.output / "rejected_labels.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "line", "reason", "raw"])
        writer.writeheader()
        writer.writerows(rejects)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
