"""
在服务器上运行: 从 COCO2017 提取 4 分类数据 (YOLO 格式)。
只提取包含 person/bicycle/motorcycle/car(truck,bus) 的图片。
图片从 zip 中按需解压，避免占用过多磁盘。
"""
import json
import os
import zipfile
from pathlib import Path
from collections import defaultdict

# COCO → BSD 类别映射
COCO_TO_BSD = {
    0:  0,   # person → 0
    1:  1,   # bicycle → 1
    3:  2,   # motorcycle → 2
    2:  3,   # car → 3 (vehicle)
    5:  3,   # bus → 3 (vehicle)
    7:  3,   # truck → 3 (vehicle)
}

ANNOTATIONS_DIR = Path("/root/autodl-tmp/BSD/datasets/coco_raw/annotations")
COCO_DIR = Path("/autodl-pub/data/COCO2017")
OUTPUT_DIR = Path("/root/autodl-tmp/BSD/datasets/bsd")

def prepare_split(split_name):
    """处理 train 或 val 切分"""
    ann_file = ANNOTATIONS_DIR / f"instances_{split_name}2017.json"
    zip_file = COCO_DIR / f"{split_name}2017.zip"

    print(f"\n{'='*60}")
    print(f"Processing {split_name}2017...")
    print(f"  Annotations: {ann_file}")
    print(f"  Zip: {zip_file}")

    with open(ann_file) as f:
        coco = json.load(f)

    # Build image_id → file_name mapping
    images = {img['id']: img for img in coco['images']}
    print(f"  Total images: {len(images)}")

    # Find images containing target classes
    target_img_ids = set()
    img_annotations = defaultdict(list)  # image_id → [annotations]

    for ann in coco['annotations']:
        cat_id = ann['category_id']
        if cat_id in COCO_TO_BSD:
            target_img_ids.add(ann['image_id'])
            img_annotations[ann['image_id']].append(ann)

    print(f"  Images with target classes: {len(target_img_ids)}")

    out_img_dir = OUTPUT_DIR / split_name / "images"
    out_lbl_dir = OUTPUT_DIR / split_name / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Open zip once
    processed = 0
    skipped = 0
    with zipfile.ZipFile(zip_file, 'r') as zf:
        # List all files in zip
        zip_names = set(zf.namelist())

        for img_id in target_img_ids:
            img_info = images[img_id]
            fname = img_info['file_name']  # e.g. "000000000139.jpg"
            zip_path = f"{split_name}2017/{fname}"

            if zip_path not in zip_names:
                skipped += 1
                continue

            # Extract image
            dst_img = out_img_dir / fname
            if not dst_img.exists():
                zf.extract(zip_path, path=str(out_img_dir.parent))
                # Move from subdir to images/
                extracted = out_img_dir.parent / zip_path
                if extracted.exists():
                    extracted.rename(dst_img)
                # Cleanup empty dir
                subdir = out_img_dir.parent / split_name
                if subdir.exists():
                    try:
                        subdir.rmdir()
                    except OSError:
                        pass

            # Write YOLO labels
            img_w = img_info['width']
            img_h = img_info['height']
            label_file = out_lbl_dir / f"{Path(fname).stem}.txt"

            with open(label_file, 'w') as lf:
                for ann in img_annotations[img_id]:
                    bsd_cls = COCO_TO_BSD[ann['category_id']]
                    x, y, w, h = ann['bbox']
                    cx = (x + w / 2.0) / img_w
                    cy = (y + h / 2.0) / img_h
                    nw = w / img_w
                    nh = h / img_h
                    # Clip to [0,1]
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    nw = max(0.0, min(1.0, nw))
                    nh = max(0.0, min(1.0, nh))
                    lf.write(f"{bsd_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

            processed += 1
            if processed % 5000 == 0:
                print(f"  Progress: {processed}/{len(target_img_ids)}")

    print(f"  Done: {processed} images extracted, {skipped} skipped")

    # Cleanup extracted subdirectory
    subdir = out_img_dir.parent / f"{split_name}2017"
    if subdir.exists():
        import shutil
        shutil.rmtree(subdir)

    # Print stats
    # Count by class
    class_counts = defaultdict(int)
    for lbl_file in out_lbl_dir.glob("*.txt"):
        with open(lbl_file) as f:
            for line in f:
                cls_id = int(line.split()[0])
                class_counts[cls_id] += 1

    names = {0: "person", 1: "bicycle", 2: "motorcycle", 3: "vehicle"}
    print(f"  Class distribution:")
    for cid in sorted(class_counts.keys()):
        print(f"    {names[cid]:>12}: {class_counts[cid]:>8}")

    return processed

if __name__ == "__main__":
    n_train = prepare_split("train")
    n_val = prepare_split("val")
    print(f"\n{'='*60}")
    print(f"Total: {n_train} train + {n_val} val images")
    print(f"Output: {OUTPUT_DIR}")
