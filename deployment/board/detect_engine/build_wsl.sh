#!/bin/sh
# Build live_bsd in WSL
BSD_DIR="/mnt/host/d/WORK/CODE/BSD"
BOARD_DIR="$BSD_DIR/deployment/board"
DET_DIR="$BOARD_DIR/detect_engine"
ALM_DIR="$BOARD_DIR/alarm_engine"
COMMON_DIR="$BOARD_DIR/common"

TC_DIR="$BSD_DIR/_toolchain/openwrt-sdk-19.07.7-sunxi-cortexa7_gcc-7.5.0_musl_eabi.Linux-x86_64"
TC_PREFIX="$TC_DIR/staging_dir/toolchain-arm_cortex-a7+neon-vfpv4_gcc-7.5.0_musl_eabi"
CC="$TC_PREFIX/bin/arm-openwrt-linux-muslgnueabi-gcc"
SYSROOT="$TC_PREFIX/arm-openwrt-linux-muslgnueabi/sysroot"

CFLAGS="-O2 -Wall -std=c11 -fPIC \
  -I$DET_DIR \
  -I$ALM_DIR \
  -I$COMMON_DIR \
  -I$TC_PREFIX/include \
  --sysroot=$SYSROOT"

LDFLAGS="-L$DET_DIR -Wl,-rpath,/usr/lib -Wl,--allow-shlib-undefined"

echo "=== Building live_bsd ==="
echo "CC: $CC"

set -e

echo "--- yolo_decode.o ---"
$CC $CFLAGS -c "$DET_DIR/yolo_decode.c" -o "$DET_DIR/yolo_decode.o"

echo "--- detect_engine.o ---"
$CC $CFLAGS -c "$DET_DIR/detect_engine.c" -o "$DET_DIR/detect_engine.o"

echo "--- preprocess.o ---"
$CC $CFLAGS -c "$DET_DIR/preprocess.c" -o "$DET_DIR/preprocess.o"

echo "--- live_bsd.o ---"
$CC $CFLAGS -c "$DET_DIR/live_bsd.c" -o "$DET_DIR/live_bsd.o"

echo "--- alarm_engine.o ---"
$CC $CFLAGS -c "$ALM_DIR/alarm_engine.c" -o "$DET_DIR/alarm_engine.o"

echo "--- zone_mgr.o ---"
$CC $CFLAGS -c "$ALM_DIR/zone_mgr.c" -o "$DET_DIR/zone_mgr.o"

echo "--- tracker.o ---"
$CC $CFLAGS -c "$ALM_DIR/tracker.c" -o "$DET_DIR/tracker.o"

echo "--- linking live_bsd ---"
$CC $CFLAGS \
  -o "$DET_DIR/live_bsd" \
  "$DET_DIR/detect_engine.o" \
  "$DET_DIR/preprocess.o" \
  "$DET_DIR/yolo_decode.o" \
  "$DET_DIR/alarm_engine.o" \
  "$DET_DIR/zone_mgr.o" \
  "$DET_DIR/tracker.o" \
  "$DET_DIR/live_bsd.o" \
  $LDFLAGS \
  -l:libawnn_full.so \
  -l:libVIPuser.so \
  -l:libVIPlite.so \
  -lstdc++ -lm -lpthread -lrt

echo ""
echo "=== SUCCESS ==="
ls -la "$DET_DIR/live_bsd"
file "$DET_DIR/live_bsd"
