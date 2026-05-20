"""Deploy BSD engine to V853 board and run end-to-end test."""
import subprocess
import os
import numpy as np
from PIL import Image

ADB = r'D:\WORK\TOOL\AllwinnertechPhoeniSuit (1)\AllwinnertechPhoeniSuitRelease20201225\adb.exe'

def adb(cmd, timeout=30):
    full = f'"{ADB}" {cmd}'
    r = subprocess.run(full, shell=True, capture_output=True, timeout=timeout)
    return r.stdout.decode('utf-8', errors='replace') + r.stderr.decode('utf-8', errors='replace')

print('=== BSD Engine Board Test ===\n')
print('ADB devices:')
print(adb('devices').strip())

BIN_DIR = r'd:\WORK\CODE\BSD\artifacts\board_bin'
NB_PATH  = r'd:\WORK\CODE\BSD\artifacts\v4_model\bsd_v4_640.nb'

# Use the SAME test input that test_v4.c used (known good)
RAW_PATH = r'd:\WORK\CODE\BSD\artifacts\v4_model\test_input.raw'
if not os.path.exists(RAW_PATH):
    print(f'Creating test input from calibration image...')
    CALIB_DIR = r'd:\WORK\CODE\BSD\artifacts\v4_model\calib_images'
    imgs = [f for f in os.listdir(CALIB_DIR) if f.endswith('.jpg')]
    img = Image.open(os.path.join(CALIB_DIR, imgs[0])).convert('RGB')
    arr = np.array(img, dtype=np.uint8)
    # Create BGR interleaved version for the engine (640x640 camera)
    bgr = arr[:, :, ::-1].copy()
    bgr.tofile(RAW_PATH)
    print(f'  Created: {RAW_PATH} ({os.path.getsize(RAW_PATH)} bytes)')
else:
    # The existing test_input.raw is RGB planar. Convert to BGR interleaved for engine.
    print(f'Using existing test_input.raw, converting to BGR interleaved...')
    rgb_planar = np.fromfile(RAW_PATH, dtype=np.uint8).reshape(3, 640, 640)
    rgb = rgb_planar.transpose(1, 2, 0)  # HWC RGB
    bgr = rgb[:, :, ::-1].copy()  # BGR
    BGR_PATH = r'd:\WORK\CODE\BSD\artifacts\board_bin\test_bgr_640x640.raw'
    bgr.tofile(BGR_PATH)
    RAW_PATH = BGR_PATH
    print(f'  BGR interleaved: {RAW_PATH} ({os.path.getsize(RAW_PATH)} bytes)')

# Push to board
print('\n--- Pushing to board ---')
print('Pushing bsd_detect...')
print(adb(f'push "{BIN_DIR}\\bsd_detect" /mnt/UDISK/bsd_detect'))
print('Pushing NB...')
print(adb(f'push "{NB_PATH}" /mnt/UDISK/bsd_v4_640.nb'))
print('Pushing test frame...')
print(adb(f'push "{RAW_PATH}" /mnt/UDISK/test_frame.raw'))
print('Chmod...')
print(adb('shell chmod +x /mnt/UDISK/bsd_detect'))

print('\nFiles on board:')
print(adb('shell ls -lh /mnt/UDISK/bsd_detect /mnt/UDISK/bsd_v4_640.nb /mnt/UDISK/test_frame.raw'))

# Test 1: 640x640 camera (no letterbox scaling)
print('\n=== Test 1: 640x640 camera ===')
result = adb('shell "cat /mnt/UDISK/test_frame.raw | /mnt/UDISK/bsd_detect /mnt/UDISK/bsd_v4_640.nb 640 640 0.3 0.45"', timeout=60)
print(result)

# Test 2: 1920x1080 camera (letterbox scaling)
# Create a 1920x1080 test frame: embed the 640x640 BGR image centered in gray
print('\n=== Test 2: 1920x1080 camera ===')
rgb_planar = np.fromfile(r'd:\WORK\CODE\BSD\artifacts\v4_model\test_input.raw', dtype=np.uint8).reshape(3, 640, 640)
rgb = rgb_planar.transpose(1, 2, 0)
bgr_640 = rgb[:, :, ::-1].copy()

cam_w, cam_h = 1920, 1080
full = np.full((cam_h, cam_w, 3), 114, dtype=np.uint8)
off_x = (cam_w - 640) // 2
off_y = (cam_h - 640) // 2
full[off_y:off_y+640, off_x:off_x+640] = bgr_640

FULL_PATH = r'd:\WORK\CODE\BSD\artifacts\board_bin\test_1920x1080.raw'
full.tofile(FULL_PATH)
print(f'  Created: {FULL_PATH} ({os.path.getsize(FULL_PATH)} bytes)')

print('Pushing 1920x1080 frame...')
print(adb(f'push "{FULL_PATH}" /mnt/UDISK/test_1920x1080.raw'))

print('\nBoard test (1920x1080):')
result = adb('shell "cat /mnt/UDISK/test_1920x1080.raw | /mnt/UDISK/bsd_detect /mnt/UDISK/bsd_v4_640.nb 1920 1080 0.3 0.45"', timeout=60)
print(result)

print('\n=== DONE ===')
