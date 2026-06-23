#!/bin/sh
set -e
BSD_DIR="/mnt/host/d/WORK/CODE/BSD"
DET_DIR="$BSD_DIR/deployment/board/detect_engine"

TC_DIR="$BSD_DIR/_toolchain/openwrt-sdk-19.07.7-sunxi-cortexa7_gcc-7.5.0_musl_eabi.Linux-x86_64"
TC_PREFIX="$TC_DIR/staging_dir/toolchain-arm_cortex-a7+neon-vfpv4_gcc-7.5.0_musl_eabi"
CC="$TC_PREFIX/bin/arm-openwrt-linux-muslgnueabi-gcc"
SYSROOT="$TC_PREFIX/arm-openwrt-linux-muslgnueabi/sysroot"

CFLAGS="-O2 -Wall -std=c11 -fPIC \
  -I$DET_DIR \
  -I$BSD_DIR/deployment/board/alarm_engine \
  -I$BSD_DIR/deployment/board/common \
  -I$TC_PREFIX/include \
  --sysroot=$SYSROOT"

LDFLAGS="-L$DET_DIR -Wl,-rpath,/usr/lib -Wl,--allow-shlib-undefined"

echo "=== Compiling ==="
$CC $CFLAGS -c "$DET_DIR/detect_engine.c" -o "$DET_DIR/detect_engine.o"
$CC $CFLAGS -c "$DET_DIR/yolo_decode.c" -o "$DET_DIR/yolo_decode.o"
$CC $CFLAGS -c "$DET_DIR/preprocess.c" -o "$DET_DIR/preprocess.o"
$CC $CFLAGS -c "$DET_DIR/test_npu_direct.c" -o "$DET_DIR/test_npu_direct.o"

echo "=== Linking test_npu_direct ==="
$CC $CFLAGS -o "$DET_DIR/test_npu_direct" \
  "$DET_DIR/detect_engine.o" \
  "$DET_DIR/preprocess.o" \
  "$DET_DIR/yolo_decode.o" \
  "$DET_DIR/test_npu_direct.o" \
  $LDFLAGS \
  -l:libawnn_full.so -l:libVIPuser.so -l:libVIPlite.so -lstdc++ -lm -lpthread -lrt

echo ""
echo "=== SUCCESS ==="
ls -la "$DET_DIR/test_npu_direct"
file "$DET_DIR/test_npu_direct"
