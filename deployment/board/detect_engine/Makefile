# Cross-compile for Allwinner V853 (ARM32 musl)
TOOLCHAIN = $(HOME)/tina-v853-100ask/prebuilt/gcc/linux-x86/arm/toolchain-sunxi-musl/toolchain
CC        = $(TOOLCHAIN)/bin/arm-openwrt-linux-muslgnueabi-gcc
AWNNDIR   = $(HOME)/tina-v853-100ask/package/allwinner/libawnn_full
VIPDIR    = $(HOME)/tina-v853-100ask/package/allwinner/libsdk-viplite-driver

CFLAGS  = -O2 -Wall -std=c11 -fPIC -I.. -I. -I../alarm_engine -I$(AWNNDIR)/sdk/include
LIBS    = -L$(AWNNDIR)/sdk/library/musl -L$(VIPDIR)/sdk_release/library/musl \
          -Wl,--start-group -l:libawnn_full.a -l:libVIPuser.a -l:libVIPlite.a -Wl,--end-group \
          -lstdc++ -lm -lpthread -lrt

# Detect engine objects
DET_OBJS = detect_engine.o preprocess.o yolo_decode.o

# Alarm engine objects (built from ../alarm_engine/)
ALM_DIR  = ../alarm_engine
ALM_OBJS = alarm_engine.o zone_mgr.o tracker.o

# Shared library
SO_TARGET = libdetect_engine.so

# Standalone binary
BIN_TARGET = bsd_detect

all: $(SO_TARGET) $(BIN_TARGET)

$(SO_TARGET): $(DET_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -shared $(LIBS)

$(BIN_TARGET): $(DET_OBJS) $(ALM_OBJS) main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Alarm engine objects built from ../alarm_engine/
alarm_engine.o: $(ALM_DIR)/alarm_engine.c
	$(CC) $(CFLAGS) -c $< -o $@

zone_mgr.o: $(ALM_DIR)/zone_mgr.c
	$(CC) $(CFLAGS) -c $< -o $@

tracker.o: $(ALM_DIR)/tracker.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(DET_OBJS) $(ALM_OBJS) main.o $(SO_TARGET) $(BIN_TARGET)

.PHONY: all clean
