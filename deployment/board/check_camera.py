"""Check V853 camera capabilities via ADB."""
import subprocess
import sys

ADB = r'D:\WORK\TOOL\AllwinnertechPhoeniSuit (1)\AllwinnertechPhoeniSuitRelease20201225\adb.exe'

def adb(cmd, timeout=15):
    r = subprocess.run(f'"{ADB}" {cmd}', shell=True, capture_output=True, timeout=timeout)
    out = r.stdout.decode('utf-8', errors='replace')
    err = r.stderr.decode('utf-8', errors='replace')
    # Strip ANSI color codes
    import re
    out = re.sub(r'\x1b\[[0-9;]*m', '', out)
    err = re.sub(r'\x1b\[[0-9;]*m', '', err)
    return out + err

print("=== /dev/video devices ===")
print(adb('shell ls -la /dev/video*'))
print()

print("=== /dev/video0 info ===")
print(adb('shell "cat /sys/class/video4linux/video0/name 2>/dev/null; echo ---; cat /sys/class/video4linux/video0/dev 2>/dev/null"'))
print()

# Check if v4l2-ctl exists on board
print("=== v4l2-ctl check ===")
print(adb('shell "which v4l2-ctl 2>/dev/null || echo not_found"'))
print()

# Try media-ctl
print("=== media-ctl check ===")
print(adb('shell "which media-ctl 2>/dev/null || echo not_found"'))
print()

# Check kernel messages for camera
print("=== dmesg camera lines ===")
print(adb('shell "dmesg | grep -i -E \"camera|sensor|vin|vipp|vi\" | tail -30"'))
print()

# List /dev/media*
print("=== /dev/media* ===")
print(adb('shell "ls -la /dev/media* 2>/dev/null || echo no_media_devices"'))
print()

# Check available V4L2 formats via ioctl... let's try a simple capture
print("=== Frame buffer size check ===")
print(adb('shell "cat /sys/class/video4linux/video0/format 2>/dev/null || echo no_format_sysfs"'))
print()

# Check frame sizes
for f in ['video0', 'video4', 'video8', 'video12']:
    r = adb(f'shell "ls /sys/class/video4linux/{f}/ 2>/dev/null"')
    if r.strip():
        print(f"=== /sys/class/video4linux/{f}/ files ===")
        print(r[:500])
