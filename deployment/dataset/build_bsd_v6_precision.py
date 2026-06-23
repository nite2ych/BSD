"""
Build BSD v6 precision dataset from v5 plus reviewed hard-negative/corrected sets.

Reviewed extra sets are passed as:
  --extra-set name:/path/to/images:/path/to/labels

Every image in an extra set must have a label file. Empty label files are valid
and are used for hard-negative images.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


CLASS_NAMES = ["person", "bicycle", "motorcycle", "vehicle"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class ImageSet:
    name: str
    image_dir: Path
    label_dir: Path


def parse_extra_set(raw: str) -> ImageSet:
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("extra set must be name:image_dir:label_dir")
    return ImageSet(parts[0], Path(parts[1]), Path(parts[2]))


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


def read_label_stats(path: Path) -> tuple[Counter, int]:
    counts: Counter = Counter()
    lines = 0
    if not path.exists():
        raise FileNotFoundError(path)
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        if 0 <= cls < len(CLASS_NAMES):
            counts[CLASS_NAMES[cls]] += 1
            lines += 1
    return counts, lines


def yaml_value(root: Path, value: Path) -> str:
    try:
        return value.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return value.as_posix()


def write_yaml(path: Path, train_images: Path, val_images: Path) -> None:
    root = path.parent
    names = ", ".join(CLASS_NAMES)
    path.write_text(
        "\n".join(
            [
                f"path: {root.as_posix()}",
                f"train: {yaml_value(root, train_images)}",
                f"val: {yaml_value(root, val_images)}",
                "nc: 4",
                f"names: [{names}]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def copy_split(src_images: Path, src_labels: Path, dst_split: Path, mode: str, prefix: str = "") -> dict[str, object]:
    summary = {"images": 0, "empty_labels": 0, "object_counts": Counter(), "missing_labels": []}
    for image in sorted(p for p in src_images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        label = src_labels / f"{image.stem}.txt"
        if not label.exists():
            summary["missing_labels"].append(str(label))
            continue
        stem = f"{prefix}{image.stem}" if prefix else image.stem
        image_dst = dst_split / "images" / f"{stem}{image.suffix.lower()}"
        label_dst = dst_split / "labels" / f"{stem}.txt"
        link_or_copy(image, image_dst, mode)
        label_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(label, label_dst)
        counts, line_count = read_label_stats(label)
        summary["images"] += 1
        if line_count == 0:
            summary["empty_labels"] += 1
        summary["object_counts"].update(counts)
    if summary["missing_labels"]:
        raise SystemExit(f"missing labels in {src_images}: {summary['missing_labels'][:10]}")
    summary["object_counts"] = dict(summary["object_counts"])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("/root/autodl-tmp/BSD/datasets/bsd_v5_balanced"))
    parser.add_argument("--coco-val", type=Path, default=Path("/root/autodl-tmp/BSD/datasets/bsd/val/images"))
    parser.add_argument("--output", type=Path, default=Path("/root/autodl-tmp/BSD/datasets/bsd_v6_precision"))
    parser.add_argument("--extra-set", action="append", type=parse_extra_set, default=[])
    parser.add_argument("--link-mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    args = parser.parse_args()

    if args.output.exists():
        shutil.rmtree(args.output)
    for split in ["train", "bdd_proxy_val"]:
        (args.output / split / "images").mkdir(parents=True, exist_ok=True)
        (args.output / split / "labels").mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "base": str(args.base),
        "output": str(args.output),
        "extra_sets": [],
    }
    train_summaries = {}
    train_summaries["base_v5_train"] = copy_split(
        args.base / "train/images",
        args.base / "train/labels",
        args.output / "train",
        args.link_mode,
    )
    copy_split(
        args.base / "bdd_proxy_val/images",
        args.base / "bdd_proxy_val/labels",
        args.output / "bdd_proxy_val",
        args.link_mode,
    )

    for extra in args.extra_set:
        prefix = f"{extra.name}_"
        summary = copy_split(extra.image_dir, extra.label_dir, args.output / "train", args.link_mode, prefix=prefix)
        train_summaries[extra.name] = summary
        manifest["extra_sets"].append(
            {"name": extra.name, "image_dir": str(extra.image_dir), "label_dir": str(extra.label_dir), "summary": summary}
        )

    write_yaml(args.output / "dataset.yaml", args.output / "train/images", args.coco_val)
    write_yaml(args.output / "bdd_proxy_val.yaml", args.output / "train/images", args.output / "bdd_proxy_val/images")
    manifest["train_summaries"] = train_summaries
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
