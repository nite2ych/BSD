# Cross-compile for Allwinner V853 (ARM32 musl)
TOOLCHAIN = $(HOME)/tina-v853-100ask/prebuilt/gcc/linux-x86/arm/toolchain-sunxi-musl/toolchain
CC        = $(TOOLCHAIN)/bin/arm-openwrt-linux-muslgnueabi-gcc
AWNNDIR   = $(HOME)/tina-v853-100ask/package/allwinner/libawnn_full
VIPDIR    = $(HOME)/tina-v853-100ask/package/allwinner/libsdk-viplite-driver

CFLAGS  = -O2 -Wall -std=c11 -fPIC -I.. -I.
LDFLAGS = -shared -L$(AWNNDIR)/sdk/library/musl -L$(VIPDIR)/sdk_release/library/musl \
          -Wl,--start-group -l:libawnn_full.a -l:libVIPuser.a -l:libVIPlite.a -Wl,--end-group \
          -lstdc++ -lm -lpthread -lrt

OBJS   = detect_engine.o preprocess.o yolo_decode.o
TARGET = libdetect_engine.so

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
