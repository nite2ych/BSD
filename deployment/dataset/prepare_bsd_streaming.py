"""
Streaming COCO dataset preparation using ijson.
Memory-efficient: parses JSON incrementally without loading all into RAM.
"""
import ijson
import os
from pathlib import Path
from collections import defaultdict

COCO_TO_BSD = {
    1: 0,  # person    → 0
    2: 1,  # bicycle   → 1
    4: 2,  # motorcycle → 2
    3: 3,  # car       → 3 (vehicle)
    6: 3,  # bus       → 3 (vehicle)
    8: 3,  # truck     → 3 (vehicle)
}
CLS_NAMES = {0:"person", 1:"bicycle", 2:"motorcycle", 3:"vehicle"}
ANNOTATIONS_DIR = Path("/root/autodl-tmp/BSD/datasets/coco_raw/annotations")
OUTPUT_DIR = Path("/root/autodl-tmp/BSD/datasets/bsd")

def prepare_split(split_name):
    ann_file = ANNOTATIONS_DIR / f"instances_{split_name}2017.json"
    print(f"\n{'='*60}")
    print(f"Processing {split_name}2017...")

    out_img_dir = OUTPUT_DIR / split_name / "images"
    out_lbl_dir = OUTPUT_DIR / split_name / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Stream through annotations, collect image_ids with target classes
    print("Step 1: Streaming annotations...")
    img_anns = defaultdict(list)  # image_id -> [annotations]
    total_anns = 0
    kept_anns = 0

    with open(ann_file, 'rb') as f:
        for ann in ijson.items(f, 'annotations.item'):
            total_anns += 1
            if total_anns % 500000 == 0:
                print(f"  Scanned {total_anns} annotations, kept {kept_anns}...")
            cat_id = int(ann['category_id'])
            if cat_id in COCO_TO_BSD:
                kept_anns += 1
                img_anns[ann['image_id']].append({
                    'category_id': COCO_TO_BSD[cat_id],
                    'bbox': ann['bbox']
                })

    target_img_ids = set(img_anns.keys())
    print(f"  Total annotations: {total_anns}, kept: {kept_anns}")
    print(f"  Target images: {len(target_img_ids)}")

    # Step 2: Stream through images, extract info for target images
    print("Step 2: Streaming images...")
    img_info = {}  # image_id -> {file_name, width, height}
    with open(ann_file, 'rb') as f:
        for img in ijson.items(f, 'images.item'):
            if img['id'] in target_img_ids:
                img_info[img['id']] = {
                    'file_name': img['file_name'],
                    'width': img['width'],
                    'height': img['height']
                }

    print(f"  Matched images: {len(img_info)}")

    # Step 3: Write labels
    print("Step 3: Writing labels...")
    img_list_file = OUTPUT_DIR / f"{split_name}_images.txt"
    class_counts = defaultdict(int)
    processed = 0

    with open(img_list_file, 'w') as ilf:
        for img_id in target_img_ids:
            if img_id not in img_info:
                continue
            info = img_info[img_id]
            fname = info['file_name']
            w, h = info['width'], info['height']

            ilf.write(f"{fname}\n")

            lbl_path = out_lbl_dir / f"{Path(fname).stem}.txt"
            with open(lbl_path, 'w') as lf:
                for ann in img_anns[img_id]:
                    bsd_cls = ann['category_id']
                    x, y, bw, bh = ann['bbox']
                    cx = max(0.0, min(1.0, (x + bw / 2) / w))
                    cy = max(0.0, min(1.0, (y + bh / 2) / h))
                    nw = max(0.0, min(1.0, bw / w))
                    nh = max(0.0, min(1.0, bh / h))
                    lf.write(f"{bsd_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                    class_counts[bsd_cls] += 1

            processed += 1
            if processed % 5000 == 0:
                print(f"  Labels: {processed}/{len(target_img_ids)}")

    print(f"\n  Labels written: {processed}")
    for cid in sorted(class_counts.keys()):
        print(f"    {CLS_NAMES[cid]:>12}: {class_counts[cid]:>8}")
    print(f"  Image list: {img_list_file}")
    return processed

if __name__ == "__main__":
    n_train = prepare_split("train")
    n_val = prepare_split("val")
    print(f"\n{'='*60}")
    print(f"Total: {n_train} train + {n_val} val images with labels")
    print(f"Output: {OUTPUT_DIR}")
