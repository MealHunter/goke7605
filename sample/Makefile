# Makefile for test_image
# NPU目标检测示例程序 (使用xmm方式: xmedia_cl)

ifeq ($(CFG_XMEDIA_EXPORT_FLAG),)
    SDK_DIR := $(shell  cd $(CURDIR)/../../.. && /bin/pwd)
endif

include $(SDK_DIR)/build/base.mk
include $(SAMPLE_DIR)/sample_base.mk

TARGET := test_image

# 链接库 - 使用xmedia_cl替代xmedia_svp
LIBS := -lxmedia_cl $(SAMPLE_LIBS) $(SAMPLE_COMMON_LIB)

ifeq ($(TOOLCHAIN),arm-gcc12.2.0-linux)
	LIBS += -L../demo_ai/ffmpeg/lib/arm-gcc12.2.0-linux-gnueabi -lavformat -lavcodec -lavutil -lswscale -lswresample
endif

# 头文件路径
INCLUDES := $(SAMPLE_INCLUDES)
INCLUDES += -I../demo_ai/ffmpeg/include

CFLAGS := $(SAMPLE_CFLAGS) $(LIBS) $(INCLUDES) -lm -lpthread -lstdc++

SRCS := $(wildcard ./*.c)
OBJS := $(patsubst %.c, %.o, $(SRCS))

.PHONY: all clean

all: $(OBJS)
	$(AT)$(CC) -o $(TARGET) $^ $(CFLAGS)

%.o : %.c
	$(AT)$(CC) -c -o $@ $< $(CFLAGS)

clean:
	$(AT)rm -rf $(OBJS) $(TARGET)
