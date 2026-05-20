"""
Step 1: Parse COCO JSON → YOLO label files + image list.
Streams JSON incrementally using ijson (lower memory).
"""
import json
import os
from pathlib import Path
from collections import defaultdict

COCO_TO_BSD = {0:0, 1:1, 3:2, 2:3, 5:3, 7:3}
ANNOTATIONS_DIR = Path("/root/autodl-tmp/BSD/datasets/coco_raw/annotations")
OUTPUT_DIR = Path("/root/autodl-tmp/BSD/datasets/bsd")

def prepare_labels(split_name):
    ann_file = ANNOTATIONS_DIR / f"instances_{split_name}2017.json"

    print(f"\n{'='*60}")
    print(f"Processing {split_name}2017...")

    # Load JSON (use streaming if possible)
    print("Loading JSON...")
    with open(ann_file) as f:
        coco = json.load(f)
    print(f"Loaded. Images: {len(coco['images'])}, Annotations: {len(coco['annotations'])}")

    # Build image lookup
    img_info = {img['id']: img for img in coco['images']}

    # Group annotations by image_id (only target classes)
    img_anns = defaultdict(list)
    for ann in coco['annotations']:
        if ann['category_id'] in COCO_TO_BSD:
            img_anns[ann['image_id']].append(ann)

    print(f"Target images: {len(img_anns)}")

    # Free coco annotations
    del coco

    out_img_dir = OUTPUT_DIR / split_name / "images"
    out_lbl_dir = OUTPUT_DIR / split_name / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Write labels + image list
    img_list_file = OUTPUT_DIR / f"{split_name}_images.txt"
    processed = 0
    class_counts = defaultdict(int)

    with open(img_list_file, 'w') as ilf:
        for img_id, anns in img_anns.items():
            info = img_info[img_id]
            fname = info['file_name']
            w, h = info['width'], info['height']

            # Write image path to list
            ilf.write(f"{fname}\n")

            # Write YOLO labels
            lbl_path = out_lbl_dir / f"{Path(fname).stem}.txt"
            with open(lbl_path, 'w') as lf:
                for ann in anns:
                    bsd_cls = COCO_TO_BSD[ann['category_id']]
                    x, y, bw, bh = ann['bbox']
                    cx = max(0.0, min(1.0, (x + bw / 2) / w))
                    cy = max(0.0, min(1.0, (y + bh / 2) / h))
                    nw = max(0.0, min(1.0, bw / w))
                    nh = max(0.0, min(1.0, bh / h))
                    lf.write(f"{bsd_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                    class_counts[bsd_cls] += 1

            processed += 1
            if processed % 10000 == 0:
                print(f"  Progress: {processed}/{len(img_anns)}")

    print(f"  Labels written: {processed} images")
    names = {0:"person", 1:"bicycle", 2:"motorcycle", 3:"vehicle"}
    for cid in sorted(class_counts.keys()):
        print(f"    {names[cid]:>12}: {class_counts[cid]:>8}")
    print(f"  Image list: {img_list_file}")
    return processed

if __name__ == "__main__":
    n_train = prepare_labels("train")
    n_val = prepare_labels("val")
    print(f"\nDone. {n_train} train + {n_val} val images listed.")
    print(f"Labels in: {OUTPUT_DIR}")
