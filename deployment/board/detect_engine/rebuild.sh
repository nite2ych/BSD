#!/bin/sh
set -e
cd ~/bsd_build

TOOLCHAIN=$HOME/tina-v853-100ask/prebuilt/gcc/linux-x86/arm/toolchain-sunxi-musl/toolchain
CC=$TOOLCHAIN/bin/arm-openwrt-linux-muslgnueabi-gcc
AWNNDIR=$HOME/tina-v853-100ask/package/allwinner/libawnn_full
VIPDIR=$HOME/tina-v853-100ask/package/allwinner/libsdk-viplite-driver

CFLAGS="-O2 -Wall -std=c11 -I. -I.. -I$AWNNDIR/sdk/include"
LIBS="-L$AWNNDIR/sdk/library/musl -L$VIPDIR/sdk_release/library/musl -Wl,--start-group -l:libawnn_full.a -l:libVIPuser.a -l:libVIPlite.a -Wl,--end-group -lstdc++ -lm -lpthread -lrt"

echo "=== Recompiling yolo_decode.o ==="
$CC $CFLAGS -c yolo_decode.c -o yolo_decode.o -w
echo "OK"

echo "=== Relinking test_npu_direct ==="
$CC $CFLAGS -o test_npu_direct detect_engine.o preprocess.o yolo_decode.o test_npu_direct.o $LIBS -w
echo "OK"

ls -la test_npu_direct
file test_npu_direct
