# Makefile for libxm_image_infer.a

ifeq ($(CFG_XMEDIA_EXPORT_FLAG),)
    SDK_DIR := $(shell cd $(CURDIR)/../../.. && /bin/pwd)
endif

include $(SDK_DIR)/build/base.mk
include $(SAMPLE_DIR)/sample_base.mk

LIB_NAME := libxm_image_infer.a
DEMO_NAME := test_image_demo

BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj
LIB_DIR := lib
BIN_DIR := bin

LIB_SRCS := $(wildcard src/*.c)
LIB_OBJS := $(patsubst src/%.c,$(OBJ_DIR)/src/%.o,$(LIB_SRCS))
DEMO_SRCS := $(wildcard examples/*.c)
DEMO_OBJS := $(patsubst examples/%.c,$(OBJ_DIR)/examples/%.o,$(DEMO_SRCS))

INCLUDES := $(SAMPLE_INCLUDES)
INCLUDES += -I./include
INCLUDES += -I../demo_ai/ffmpeg/include

LIBS := -lxmedia_cl $(SAMPLE_LIBS) $(SAMPLE_COMMON_LIB) -lm -lpthread -lstdc++

ifeq ($(TOOLCHAIN),arm-gcc12.2.0-linux)
	LIBS += -L../demo_ai/ffmpeg/lib/arm-gcc12.2.0-linux-gnueabi -lavformat -lavcodec -lavutil -lswscale -lswresample
endif

CFLAGS := $(SAMPLE_CFLAGS) $(INCLUDES)

.PHONY: all clean dirs

all: dirs $(LIB_DIR)/$(LIB_NAME) $(BIN_DIR)/$(DEMO_NAME)

dirs:
	$(AT)mkdir -p $(OBJ_DIR)/src $(OBJ_DIR)/examples $(LIB_DIR) $(BIN_DIR)

$(OBJ_DIR)/src/%.o: src/%.c
	$(AT)$(CC) -c -o $@ $< $(CFLAGS)

$(OBJ_DIR)/examples/%.o: examples/%.c
	$(AT)$(CC) -c -o $@ $< $(CFLAGS)

$(LIB_DIR)/$(LIB_NAME): $(LIB_OBJS)
	$(AT)$(AR) rcs $@ $^

$(BIN_DIR)/$(DEMO_NAME): $(DEMO_OBJS) $(LIB_DIR)/$(LIB_NAME)
	$(AT)$(CC) -o $@ $(DEMO_OBJS) -L$(LIB_DIR) -lxm_image_infer $(LIBS)

clean:
	$(AT)rm -rf $(BUILD_DIR) $(LIB_DIR) $(BIN_DIR)
