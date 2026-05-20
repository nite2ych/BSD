# deployment/board/alarm_engine/Makefile
# Cross-compile to libalarm_engine.so (ARM32 musl)

TOOLCHAIN = ~/tina-v853-100ask/prebuilt/gcc/linux-x86/arm/toolchain-sunxi-musl/toolchain
CC        = $(TOOLCHAIN)/bin/arm-openwrt-linux-muslgnueabi-gcc

CFLAGS  = -O2 -Wall -std=c11 -fPIC -I.. -I.
LDFLAGS = -shared -lstdc++ -lm -lpthread -lrt

OBJS   = alarm_engine.o zone_mgr.o tracker.o
TARGET = libalarm_engine.so

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
