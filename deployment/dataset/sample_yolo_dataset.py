"""
Sample a YOLO-format dataset into a smaller balanced extra set.

This is intended for public supplemental data where one class dominates. For
BSD, keep all images containing weak classes such as person/bicycle/motorcycle,
then cap images that contain only vehicle.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
BSD_NAMES = ["person", "bicycle", "motorcycle", "vehicle"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-if-any-class", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--cap-only-class", type=int, default=3)
    parser.add_argument("--max-only-class-images", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--link-mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
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


def read_classes(label: Path) -> list[int]:
    classes: list[int] = []
    if not label.exists():
        return classes
    for raw in label.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw.split()
        if not parts:
            continue
        try:
            cls_id = int(float(parts[0]))
        except ValueError:
            continue
        classes.append(cls_id)
    return classes


def count_objects(labels: Iterable[Path]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for label in labels:
        for cls_id in read_classes(label):
            if 0 <= cls_id < len(BSD_NAMES):
                counts[BSD_NAMES[cls_id]] += 1
    return counts


def main() -> int:
    args = parse_args()
    if args.output.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists, pass --overwrite to replace: {args.output}")
        shutil.rmtree(args.output)
    out_images = args.output / "images"
    out_labels = args.output / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    keep_classes = set(args.keep_if_any_class)
    keep: list[Path] = []
    only_class: list[Path] = []
    skipped_empty = 0
    skipped_mixed_without_keep = 0

    for image in sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        label = args.labels / f"{image.stem}.txt"
        classes = read_classes(label)
        class_set = set(classes)
        if not classes:
            skipped_empty += 1
            continue
        if class_set & keep_classes:
            keep.append(image)
        elif class_set == {args.cap_only_class}:
            only_class.append(image)
        else:
            skipped_mixed_without_keep += 1

    rng.shuffle(only_class)
    selected_only = only_class[: max(0, args.max_only_class_images)]
    selected = sorted(keep + selected_only)

    for image in selected:
        label = args.labels / f"{image.stem}.txt"
        link_or_copy(image, out_images / image.name, args.link_mode)
        shutil.copy2(label, out_labels / label.name)

    manifest = {
        "source_images": str(args.images),
        "source_labels": str(args.labels),
        "output": str(args.output),
        "filters": {
            "keep_if_any_class": sorted(keep_classes),
            "cap_only_class": args.cap_only_class,
            "max_only_class_images": args.max_only_class_images,
            "seed": args.seed,
        },
        "source_counts": {
            "keep_class_images": len(keep),
            "only_class_candidates": len(only_class),
            "selected_only_class_images": len(selected_only),
            "skipped_empty": skipped_empty,
            "skipped_mixed_without_keep": skipped_mixed_without_keep,
        },
        "images_written": len(selected),
        "object_counts": dict(count_objects(out_labels.glob("*.txt"))),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
