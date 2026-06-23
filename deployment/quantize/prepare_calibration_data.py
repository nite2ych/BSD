"""Prepare calibration dataset for pegasus quantization.

Takes real BSD camera frames (JPEG + BGR raw), applies letterbox and
augmentation to create a representative calibration set.
"""
import os, sys, glob
import numpy as np
from PIL import Image, ImageEnhance

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts', 'calib_camera')
NUM_AUGMENTED = 150  # total calibration images to generate


def letterbox_rgb(img, target=640):
    """RGB PIL image -> letterboxed 640x640 RGB PIL image."""
    w, h = img.size
    scale = target / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    pad_w = (target - new_w) // 2
    pad_h = (target - new_h) // 2
    result = Image.new('RGB', (target, target), (114, 114, 114))
    result.paste(resized, (pad_w, pad_h))
    return result


def bgr_raw_to_rgb(raw_path, w, h):
    """Load BGR interleaved raw, convert to RGB PIL image."""
    bgr = np.fromfile(raw_path, dtype=np.uint8).reshape(h, w, 3)
    rgb = bgr[:, :, ::-1].copy()  # BGR -> RGB
    return Image.fromarray(rgb)


def augment(img, idx):
    """Generate augmented variants of an image."""
    variants = []

    # Original
    variants.append(('orig', img))

    # Brightness variations
    for factor in [0.7, 0.85, 1.15, 1.3]:
        enh = ImageEnhance.Brightness(img)
        variants.append((f'b{int(factor*100)}', enh.enhance(factor)))

    # Contrast variations
    for factor in [0.7, 0.85, 1.15, 1.3]:
        enh = ImageEnhance.Contrast(img)
        variants.append((f'c{int(factor*100)}', enh.enhance(factor)))

    # Color variations (simulate different white balance)
    for factor in [0.8, 1.2]:
        enh = ImageEnhance.Color(img)
        variants.append((f'color{int(factor*100)}', enh.enhance(factor)))

    return variants


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    sources = []

    # Source 1: JPEG frames from board camera (already captured)
    jpg_dir = OUT_DIR
    for f in sorted(glob.glob(os.path.join(jpg_dir, '*.jpg'))):
        img = Image.open(f).convert('RGB')
        # These are direct camera captures, probably need letterbox
        if img.size != (640, 640):
            img = letterbox_rgb(img)
        sources.append((os.path.basename(f).replace('.jpg', ''), img))
        print(f"Loaded JPEG: {f} -> {img.size}")

    # Source 2: BGR raw frames
    artifacts_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts')
    bgr_files = [
        (os.path.join(artifacts_dir, 'board_frame_new.raw'), 640, 360),
        (os.path.join(artifacts_dir, 'board_frame_f30.raw'), 640, 360),
        (os.path.join(artifacts_dir, 'frame_bgr.raw'), 640, 360),
        (os.path.join(artifacts_dir, 'frame_bgr_wb.raw'), 640, 360),
    ]
    for path, w, h in bgr_files:
        if os.path.exists(path):
            img = bgr_raw_to_rgb(path, w, h)
            img = letterbox_rgb(img)
            name = os.path.basename(path).replace('.raw', '')
            sources.append((name, img))
            print(f"Loaded BGR: {path} -> {img.size}")

    # Generate augmented calibration set
    print(f"\nGenerating {NUM_AUGMENTED} calibration images from {len(sources)} sources...")

    count = 0
    for src_name, src_img in sources:
        variants = augment(src_img, count)
        for var_name, var_img in variants:
            if count >= NUM_AUGMENTED:
                break
            fname = f"calib_{count:04d}_{src_name}_{var_name}.jpg"
            var_img.save(os.path.join(OUT_DIR, fname), quality=95)
            count += 1
        if count >= NUM_AUGMENTED:
            break

    # Remove original JPEGs that are not calibration images
    for f in glob.glob(os.path.join(OUT_DIR, 'frame_*.jpg')):
        os.remove(f)

    print(f"Done: {count} calibration images saved to {OUT_DIR}")


if __name__ == '__main__':
    main()
