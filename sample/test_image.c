/*
 * Copyright (c) XMEDIA. All rights reserved.
 * test_image - NPU图像推理示例程序 (使用xmedia_cl底层API)
 * 读取一张YUV图片，使用xmm方式加载.xmm模型并进行推理
 * 
 * 编译方式:
 *   在SDK环境下 make
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <ctype.h>

#include "xmedia_cl.h"
#include "xmedia_mmz.h"
#include "xmedia_sys.h"

#include "libavformat/avformat.h"
#include "libavcodec/avcodec.h"
#include "libswscale/swscale.h"
#include "libavutil/imgutils.h"
#include "libavutil/pixfmt.h"

/* 模型文件路径 - 使用.xmm格式模型 */
#define MODEL_PATH   "/mnt/sdcard/data/neuron_network.xmm"

#define IMAGE_WIDTH  640
#define IMAGE_HEIGHT 360
#define ALIGN_BYTE   16
#define DEBUG_DUMP_BYTES 32
#define DEBUG_MAX_DIMS   8
#define DEBUG_SAMPLE_LOCATIONS 5
#define DFL_BINS 16
#define YOLO_CLASS_NUM 80
#define CANDIDATE_SCORE_THRESH 0.45f
#define NMS_IOU_THRESH 0.45f
#define MAX_CANDIDATES 6000
#define PRINT_TOPK_CANDIDATES 100
#define PERSON_CLASS_ID 0

/* 对齐宏 */
#define ALIGN_FUNC(A, ALIGN) ((((A) % (ALIGN)) == 0) ? (A) : ((A) + (ALIGN) - ((A) % (ALIGN))))

/*
 * 判断输入文件是否为 jpg/jpeg。
 * 当前示例的解码路径只实现了 JPEG -> RGB24，
 * 因此入口处先做一次简单的扩展名过滤，避免后续流程报出不必要的 FFmpeg 错误。
 */
static xmedia_bool is_jpeg_file(const char *filename)
{
    const char *ext = strrchr(filename, '.');

    if (ext == NULL) {
        return XMEDIA_FALSE;
    }

    if (tolower((unsigned char)ext[1]) == 'j' &&
        tolower((unsigned char)ext[2]) == 'p' &&
        tolower((unsigned char)ext[3]) == 'g' &&
        ext[4] == '\0') {
        return XMEDIA_TRUE;
    }

    if (tolower((unsigned char)ext[1]) == 'j' &&
        tolower((unsigned char)ext[2]) == 'p' &&
        tolower((unsigned char)ext[3]) == 'e' &&
        tolower((unsigned char)ext[4]) == 'g' &&
        ext[5] == '\0') {
        return XMEDIA_TRUE;
    }

    return XMEDIA_FALSE;
}

/*
 * 获取文件大小，仅用于启动日志打印，方便确认板端实际读取到的是哪张图片。
 */
static xmedia_s64 get_file_size(const char *filename)
{
    struct stat file_stat;

    if (stat(filename, &file_stat) != 0) {
        return -1;
    }

    return (xmedia_s64)file_stat.st_size;
}

/*
 * 打印一小段十六进制数据，用于快速核对输入/输出缓存是否“看起来合理”。
 * 这里只打印前 DEBUG_DUMP_BYTES 个字节，避免日志量过大。
 */
static xmedia_void dump_bytes_hex(const char *label, const xmedia_u8 *data, xmedia_u32 size)
{
    xmedia_u32 dump_size = size > DEBUG_DUMP_BYTES ? DEBUG_DUMP_BYTES : size;
    xmedia_u32 i;

    printf("%s (%u bytes, dump %u): ", label, size, dump_size);
    for (i = 0; i < dump_size; i++) {
        printf("%02x ", data[i]);
    }
    printf("\n");
}

/*
 * 打印 tensor shape 基本信息。
 * xmedia_cl 的 shape.dims 固定长度大于实际维度，未使用的维度通常为 0，
 * 因此这里只做完整打印，后续再结合调试信息分析真实布局。
 */
static xmedia_void print_tensor_shape(const xmedia_cl_tensor *tensor)
{
    xmedia_u32 i;
    xmedia_u32 dim_count = sizeof(tensor->shape.dims) / sizeof(tensor->shape.dims[0]);

    if (dim_count > DEBUG_MAX_DIMS) {
        dim_count = DEBUG_MAX_DIMS;
    }

    printf("  Shape type: %d\n", tensor->shape.type);
    printf("  Shape dims: [");
    for (i = 0; i < dim_count; i++) {
        printf("%u", tensor->shape.dims[i]);
        if (i + 1 < dim_count) {
            printf(", ");
        }
    }
    printf("]\n");
}

/*
 * 根据 shape 里非 0 维度估算逻辑元素个数。
 * 这个值用于和 tensor->size 做比对，从而发现“逻辑 shape”与“实际物理内存布局”之间
 * 是否存在 padding / 对齐扩展。
 */
static xmedia_u32 get_tensor_element_count(const xmedia_cl_tensor *tensor)
{
    xmedia_u32 i;
    xmedia_u32 dim_count = sizeof(tensor->shape.dims) / sizeof(tensor->shape.dims[0]);
    xmedia_u32 total = 1;
    xmedia_bool has_nonzero_dim = XMEDIA_FALSE;

    for (i = 0; i < dim_count; i++) {
        if (tensor->shape.dims[i] == 0) {
            continue;
        }
        has_nonzero_dim = XMEDIA_TRUE;
        total *= tensor->shape.dims[i];
    }

    return has_nonzero_dim ? total : 0;
}

static xmedia_u32 get_tensor_bytes_per_element(const xmedia_cl_tensor *tensor)
{
    /* 当前调试场景只关心 FP32 / UINT8 / INT8 三种张量类型。 */
    if (tensor->shape.type == XMEDIA_CL_FP32) {
        return sizeof(xmedia_float);
    }
    if (tensor->shape.type == XMEDIA_CL_UINT8) {
        return sizeof(xmedia_u8);
    }
    if (tensor->shape.type == XMEDIA_CL_INT8) {
        return sizeof(xmedia_s8);
    }

    return 0;
}

static xmedia_u32 get_tensor_physical_width(const xmedia_cl_tensor *tensor)
{
    xmedia_u32 channels = tensor->shape.dims[1];
    xmedia_u32 height = tensor->shape.dims[2];
    xmedia_u32 logical_width = tensor->shape.dims[3];
    xmedia_u32 bytes_per_elem = get_tensor_bytes_per_element(tensor);

    /*
     * 模型导出的逻辑 shape 里 width 可能是 80/40/20，
     * 但实际输出内存往往按硬件要求做了行对齐，例如 96/64/32。
     * 这里根据 tensor size 反推“物理宽度”，后续索引必须按这个值访问，
     * 否则从第 2 行开始就会读到错误的数据。
     */
    if (channels == 0 || height == 0 || bytes_per_elem == 0) {
        return logical_width;
    }

    if (tensor->size == 0) {
        return logical_width;
    }

    return tensor->size / (channels * height * bytes_per_elem);
}

/*
 * 输出 tensor 调试摘要：量化参数、逻辑元素数、物理宽度、理论字节数与实际字节数差值。
 * 这一组日志主要用于定位 output tensor 是否带有硬件对齐 padding。
 */
static xmedia_void print_tensor_debug_summary(const xmedia_cl_tensor *tensor)
{
    xmedia_u32 elem_count = get_tensor_element_count(tensor);
    xmedia_u32 bytes_per_elem = get_tensor_bytes_per_element(tensor);
    xmedia_u32 physical_width = get_tensor_physical_width(tensor);

    printf("  Quant: scale=%.6f zp=%d\n", tensor->quant.scale, tensor->quant.zp);
    printf("  Element count(from dims): %u\n", elem_count);
    printf("  Physical width(from size): %u\n", physical_width);
    if (bytes_per_elem > 0 && elem_count > 0) {
        printf("  Expected bytes(from dims/type): %u\n", elem_count * bytes_per_elem);
        printf("  Size delta(actual-expected): %d\n", (xmedia_s32)tensor->size - (xmedia_s32)(elem_count * bytes_per_elem));
    }
}

/*
 * 统一打印输入/输出 tensor 布局，方便在板端直接看 shape、size、quant 等元信息。
 */
static xmedia_void print_tensor_layout(const char *title, xmedia_cl_tensor_info_inout *tensor_info)
{
    xmedia_u32 i;

    printf("\n========================================\n");
    printf("%s\n", title);
    printf("========================================\n");
    printf("Tensor count: %d\n", tensor_info->num);

    for (i = 0; i < tensor_info->num; i++) {
        printf("\nTensor #%u:\n", i);
        printf("  Address: %p\n", tensor_info->tensor[i].addr);
        printf("  Size: %u bytes\n", tensor_info->tensor[i].size);
        print_tensor_shape(&tensor_info->tensor[i]);
        print_tensor_debug_summary(&tensor_info->tensor[i]);
    }
}

/*
 * 使用 FFmpeg 将 JPEG 解码并缩放到模型输入尺寸，输出 RGB24 连续缓存。
 * 当前模型输入固定为 640x360、3 通道，因此这里直接生成可喂给 NPU 的基础像素数据。
 */
static xmedia_s32 decode_jpeg_to_rgb24(const char *filename, xmedia_u8 **rgb_buffer,
    xmedia_u32 dst_width, xmedia_u32 dst_height)
{
    xmedia_s32 ret = -1;
    xmedia_s32 stream_index;
    AVFormatContext *format_context = NULL;
    const AVCodec *codec = NULL;
    AVCodecContext *codec_context = NULL;
    AVPacket *packet = NULL;
    AVFrame *frame = NULL;
    AVFrame *frame_rgb = NULL;
    struct SwsContext *sws_context = NULL;
    xmedia_u8 *buffer = NULL;
    xmedia_s32 num_bytes;

    *rgb_buffer = NULL;

    if (avformat_open_input(&format_context, filename, NULL, NULL) != 0) {
        printf("Error: avformat_open_input failed for %s\n", filename);
        goto EXIT;
    }

    if (avformat_find_stream_info(format_context, NULL) < 0) {
        printf("Error: avformat_find_stream_info failed\n");
        goto EXIT;
    }

    stream_index = av_find_best_stream(format_context, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
    if (stream_index < 0 || codec == NULL) {
        printf("Error: failed to find JPEG video stream\n");
        goto EXIT;
    }

    codec_context = avcodec_alloc_context3(codec);
    if (codec_context == NULL) {
        printf("Error: avcodec_alloc_context3 failed\n");
        goto EXIT;
    }

    if (avcodec_parameters_to_context(codec_context,
        format_context->streams[stream_index]->codecpar) < 0) {
        printf("Error: avcodec_parameters_to_context failed\n");
        goto EXIT;
    }

    if (avcodec_open2(codec_context, codec, NULL) < 0) {
        printf("Error: avcodec_open2 failed\n");
        goto EXIT;
    }

    packet = av_packet_alloc();
    frame = av_frame_alloc();
    frame_rgb = av_frame_alloc();
    if (packet == NULL || frame == NULL || frame_rgb == NULL) {
        printf("Error: ffmpeg frame/packet allocation failed\n");
        goto EXIT;
    }

    num_bytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24, dst_width, dst_height, 1);
    if (num_bytes <= 0) {
        printf("Error: av_image_get_buffer_size failed\n");
        goto EXIT;
    }

    buffer = (xmedia_u8 *)malloc(num_bytes);
    if (buffer == NULL) {
        printf("Error: malloc RGB buffer failed\n");
        goto EXIT;
    }

    if (av_image_fill_arrays(frame_rgb->data, frame_rgb->linesize, buffer,
        AV_PIX_FMT_RGB24, dst_width, dst_height, 1) < 0) {
        printf("Error: av_image_fill_arrays failed\n");
        goto EXIT;
    }

    sws_context = sws_getContext(codec_context->width, codec_context->height, codec_context->pix_fmt,
        dst_width, dst_height, AV_PIX_FMT_RGB24, SWS_BILINEAR, NULL, NULL, NULL);
    if (sws_context == NULL) {
        printf("Error: sws_getContext failed\n");
        goto EXIT;
    }

    while (av_read_frame(format_context, packet) >= 0) {
        if (packet->stream_index != stream_index) {
            av_packet_unref(packet);
            continue;
        }

        if (avcodec_send_packet(codec_context, packet) < 0) {
            printf("Error: avcodec_send_packet failed\n");
            av_packet_unref(packet);
            goto EXIT;
        }
        av_packet_unref(packet);

        while (avcodec_receive_frame(codec_context, frame) >= 0) {
            printf("Decoded JPG: src=%dx%d pix_fmt=%d -> dst=%ux%u RGB24\n",
                frame->width, frame->height, frame->format, dst_width, dst_height);
            sws_scale(sws_context, (const xmedia_u8 * const *)frame->data, frame->linesize,
                0, frame->height, frame_rgb->data, frame_rgb->linesize);
            *rgb_buffer = buffer;
            buffer = NULL;
            ret = 0;
            goto EXIT;
        }
    }

    printf("Error: no decoded frame from %s\n", filename);

EXIT:
    if (buffer != NULL) {
        free(buffer);
    }
    if (sws_context != NULL) {
        sws_freeContext(sws_context);
    }
    if (frame_rgb != NULL) {
        av_frame_free(&frame_rgb);
    }
    if (frame != NULL) {
        av_frame_free(&frame);
    }
    if (packet != NULL) {
        av_packet_free(&packet);
    }
    if (codec_context != NULL) {
        avcodec_free_context(&codec_context);
    }
    if (format_context != NULL) {
        avformat_close_input(&format_context);
    }

    return ret;
}

/*
 * 将 FFmpeg 输出的 RGB24（HWC 交错排布）转换为模型需要的 CHW 平面排布。
 * 转换后内存顺序为：RRR... GGG... BBB...
 */
static xmedia_void rgb24_to_chw(const xmedia_u8 *rgb_buffer, xmedia_u8 *tensor_buffer,
    xmedia_u32 width, xmedia_u32 height)
{
    xmedia_u32 hw = width * height;
    xmedia_u32 x;
    xmedia_u32 y;

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            xmedia_u32 src_index = (y * width + x) * 3;
            xmedia_u32 dst_index = y * width + x;

            tensor_buffer[dst_index] = rgb_buffer[src_index];
            tensor_buffer[hw + dst_index] = rgb_buffer[src_index + 1];
            tensor_buffer[2 * hw + dst_index] = rgb_buffer[src_index + 2];
        }
    }
}

/*
 * 一个检测层(level)的回归头/分类头配对信息。
 * 例如 80x48 / 40x24 / 20x12 三个 level，会分别记录其 reg tensor、cls tensor 和 stride。
 */
typedef struct {
    xmedia_s32 reg_index;
    xmedia_s32 cls_index;
    xmedia_u32 height;
    xmedia_u32 width;
    xmedia_u32 stride;
} detect_level_info;

/*
 * 单个候选框的中间表示。
 * 这里保存的是已经解码到输入图像坐标系下的浮点框，后续可直接参与 NMS 和打印。
 */
typedef struct {
    xmedia_s32 class_id;
    xmedia_float score;
    xmedia_float x1;
    xmedia_float y1;
    xmedia_float x2;
    xmedia_float y2;
    xmedia_u32 stride;
    xmedia_u32 level;
} detect_candidate;

/*
 * 数值稳定版本的 sigmoid。
 * 分类输出是 logit，需要先过 sigmoid 才能得到 0~1 的分数。
 */
static xmedia_float sigmoidf_safe(xmedia_float x)
{
    if (x >= 0.0f) {
        xmedia_float z = expf(-x);
        return 1.0f / (1.0f + z);
    }

    xmedia_float z = expf(x);
    return z / (1.0f + z);
}

/*
 * 将浮点结果限制在图像合法范围内，避免框坐标越界。
 */
static xmedia_float clampf_safe(xmedia_float value, xmedia_float min_value, xmedia_float max_value)
{
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

static xmedia_float tensor_value_to_float(const xmedia_cl_tensor *tensor, xmedia_u32 channel,
    xmedia_u32 y, xmedia_u32 x)
{
    xmedia_u32 height = tensor->shape.dims[2];
    /*
     * 这里不能直接用 dims[3] 作为行跨度，必须使用物理宽度，
     * 否则带 padding 的 output tensor 会发生跨行错位。
     */
    xmedia_u32 width = get_tensor_physical_width(tensor);
    /*
     * 当前代码按 NCHW 访问 tensor：先按 channel 分块，再按 H、W 展开。
     * index 的计算必须和底层内存实际排布一致，否则分类分数和回归值都会被读错。
     */
    xmedia_u32 index = (channel * height + y) * width + x;

    if (tensor->shape.type == XMEDIA_CL_FP32) {
        const xmedia_float *data = (const xmedia_float *)tensor->addr;
        return data[index];
    }

    if (tensor->shape.type == XMEDIA_CL_UINT8) {
        const xmedia_u8 *data = (const xmedia_u8 *)tensor->addr;
        return ((xmedia_float)data[index] - tensor->quant.zp) * tensor->quant.scale;
    }

    if (tensor->shape.type == XMEDIA_CL_INT8) {
        const xmedia_s8 *data = (const xmedia_s8 *)tensor->addr;
        return ((xmedia_float)data[index] - tensor->quant.zp) * tensor->quant.scale;
    }

    return 0.0f;
}

/*
 * 将 DFL(Distribution Focal Loss) 回归输出解码为像素距离。
 * 每条边由 DFL_BINS 个离散 bin 表示，先做 softmax，再计算期望，最后乘以 stride。
 */
static xmedia_float decode_dfl_distance(const xmedia_cl_tensor *reg_tensor,
    xmedia_u32 side_index, xmedia_u32 y, xmedia_u32 x, xmedia_u32 stride)
{
    xmedia_float logits[DFL_BINS];
    xmedia_float max_logit;
    xmedia_float sum = 0.0f;
    xmedia_float expectation = 0.0f;
    xmedia_u32 k;
    xmedia_u32 base_channel = side_index * DFL_BINS;

    max_logit = tensor_value_to_float(reg_tensor, base_channel, y, x);
    for (k = 0; k < DFL_BINS; k++) {
        logits[k] = tensor_value_to_float(reg_tensor, base_channel + k, y, x);
        if (logits[k] > max_logit) {
            max_logit = logits[k];
        }
    }

    for (k = 0; k < DFL_BINS; k++) {
        logits[k] = expf(logits[k] - max_logit);
        sum += logits[k];
    }

    if (sum <= 0.0f) {
        return 0.0f;
    }

    for (k = 0; k < DFL_BINS; k++) {
        expectation += ((xmedia_float)k) * (logits[k] / sum);
    }

    return expectation * stride;
}

/* 按 score 从高到低排序，供 NMS 使用。 */
static xmedia_s32 compare_candidate_desc(const void *left, const void *right)
{
    const detect_candidate *a = (const detect_candidate *)left;
    const detect_candidate *b = (const detect_candidate *)right;

    if (a->score < b->score) {
        return 1;
    }
    if (a->score > b->score) {
        return -1;
    }
    return 0;
}

/*
 * 计算两个候选框的 IoU，用于 NMS 去重。
 */
static xmedia_float compute_iou(const detect_candidate *a, const detect_candidate *b)
{
    xmedia_float xx1 = a->x1 > b->x1 ? a->x1 : b->x1;
    xmedia_float yy1 = a->y1 > b->y1 ? a->y1 : b->y1;
    xmedia_float xx2 = a->x2 < b->x2 ? a->x2 : b->x2;
    xmedia_float yy2 = a->y2 < b->y2 ? a->y2 : b->y2;
    xmedia_float inter_w = xx2 - xx1;
    xmedia_float inter_h = yy2 - yy1;
    xmedia_float inter_area;
    xmedia_float area_a;
    xmedia_float area_b;
    xmedia_float union_area;

    if (inter_w <= 0.0f || inter_h <= 0.0f) {
        return 0.0f;
    }

    inter_area = inter_w * inter_h;
    area_a = (a->x2 - a->x1) * (a->y2 - a->y1);
    area_b = (b->x2 - b->x1) * (b->y2 - b->y1);
    union_area = area_a + area_b - inter_area;

    if (union_area <= 0.0f) {
        return 0.0f;
    }

    return inter_area / union_area;
}

/*
 * 对候选框执行按类别的 NMS。
 * 输入数组会被原地改写，返回值表示保留下来的框数量。
 */
static xmedia_u32 apply_nms(detect_candidate *candidates, xmedia_u32 candidate_count,
    xmedia_float iou_thresh)
{
    xmedia_u8 *suppressed;
    xmedia_u32 keep_count = 0;
    xmedia_u32 i;
    xmedia_u32 j;

    suppressed = (xmedia_u8 *)calloc(candidate_count, sizeof(xmedia_u8));
    if (suppressed == NULL) {
        printf("Warning: NMS suppression buffer allocation failed, skip NMS\n");
        return candidate_count;
    }

    for (i = 0; i < candidate_count; i++) {
        if (suppressed[i] != 0) {
            continue;
        }

        if (keep_count != i) {
            candidates[keep_count] = candidates[i];
        }

        for (j = i + 1; j < candidate_count; j++) {
            if (suppressed[j] != 0) {
                continue;
            }
            if (candidates[j].class_id != candidates[keep_count].class_id) {
                continue;
            }
            if (compute_iou(&candidates[keep_count], &candidates[j]) > iou_thresh) {
                suppressed[j] = 1;
            }
        }

        keep_count++;
    }

    free(suppressed);
    return keep_count;
}

/*
 * 根据 output tensor 的 shape 自动把 reg / cls tensor 配成 level。
 * 当前假设：
 *   - 回归头通道数 = DFL_BINS * 4
 *   - 分类头通道数 = YOLO_CLASS_NUM
 *   - 同一 level 的 reg/cls 具有相同的 H/W
 */
static xmedia_s32 collect_detect_levels(xmedia_cl_tensor_info_inout *output,
    detect_level_info *levels, xmedia_u32 max_levels)
{
    xmedia_u32 i;
    xmedia_u32 level_count = 0;

    for (i = 0; i < output->num; i++) {
        xmedia_cl_tensor *tensor = &output->tensor[i];
        xmedia_u32 channels = tensor->shape.dims[1];
        xmedia_u32 height = tensor->shape.dims[2];
        xmedia_u32 width = tensor->shape.dims[3];
        xmedia_u32 level_index;
        xmedia_s32 matched = -1;

        for (level_index = 0; level_index < level_count; level_index++) {
            if (levels[level_index].height == height && levels[level_index].width == width) {
                matched = (xmedia_s32)level_index;
                break;
            }
        }

        if (matched < 0) {
            if (level_count >= max_levels) {
                continue;
            }
            matched = (xmedia_s32)level_count;
            levels[level_count].reg_index = -1;
            levels[level_count].cls_index = -1;
            levels[level_count].height = height;
            levels[level_count].width = width;
            levels[level_count].stride = width == 0 ? 0 : (IMAGE_WIDTH / width);
            level_count++;
        }

        if (channels == DFL_BINS * 4) {
            levels[matched].reg_index = (xmedia_s32)i;
        } else if (channels == YOLO_CLASS_NUM) {
            levels[matched].cls_index = (xmedia_s32)i;
        }
    }

    return (xmedia_s32)level_count;
}

/*
 * 在每个 level 选几个固定位置打印原始 cls/reg 值，
 * 用于快速判断 tensor layout、padding 索引以及量化反算是否正常。
 */
static xmedia_void print_level_sample_points(const xmedia_cl_tensor *reg_tensor,
    const xmedia_cl_tensor *cls_tensor, const detect_level_info *level, xmedia_u32 level_index)
{
    xmedia_u32 sample_count = 0;
    xmedia_u32 sample_positions[DEBUG_SAMPLE_LOCATIONS][2];
    xmedia_u32 i;

    if (level->width == 0 || level->height == 0) {
        return;
    }

    sample_positions[sample_count][0] = 0;
    sample_positions[sample_count][1] = 0;
    sample_count++;

    if (sample_count < DEBUG_SAMPLE_LOCATIONS) {
        sample_positions[sample_count][0] = level->width / 2;
        sample_positions[sample_count][1] = level->height / 2;
        sample_count++;
    }

    if (sample_count < DEBUG_SAMPLE_LOCATIONS) {
        sample_positions[sample_count][0] = level->width - 1;
        sample_positions[sample_count][1] = level->height - 1;
        sample_count++;
    }

    if (sample_count < DEBUG_SAMPLE_LOCATIONS) {
        sample_positions[sample_count][0] = level->width / 4;
        sample_positions[sample_count][1] = level->height / 4;
        sample_count++;
    }

    if (sample_count < DEBUG_SAMPLE_LOCATIONS) {
        sample_positions[sample_count][0] = (level->width * 3) / 4;
        sample_positions[sample_count][1] = (level->height * 3) / 4;
        sample_count++;
    }

    printf("Level %u sample points for layout check:\n", level_index);
    for (i = 0; i < sample_count; i++) {
        xmedia_u32 x = sample_positions[i][0];
        xmedia_u32 y = sample_positions[i][1];
        xmedia_float cls0 = tensor_value_to_float(cls_tensor, 0, y, x);
        xmedia_float cls1 = tensor_value_to_float(cls_tensor, 1, y, x);
        xmedia_float reg0 = tensor_value_to_float(reg_tensor, 0, y, x);
        xmedia_float reg15 = tensor_value_to_float(reg_tensor, DFL_BINS - 1, y, x);
        xmedia_float reg16 = tensor_value_to_float(reg_tensor, DFL_BINS, y, x);

        printf("  sample[%u] xy=(%u,%u) cls0=%.4f cls1=%.4f reg[0]=%.4f reg[15]=%.4f reg[16]=%.4f\n",
            i, x, y, cls0, cls1, reg0, reg15, reg16);
    }
}

/*
 * 后处理主流程：
 *   1. 自动配对 reg/cls tensor
 *   2. 遍历每个 grid 点，找出得分最高类别
 *   3. 对 person 类执行 DFL 解码，得到边界框
 *   4. 执行 NMS
 *   5. 打印最终候选框
 *
 * 当前这份代码保留了较多调试日志，便于板端分析 stride、padding、坐标映射等问题。
 */
static xmedia_void print_candidate_boxes(xmedia_cl_tensor_info_inout *output)
{
    detect_level_info levels[3];
    detect_candidate candidates[MAX_CANDIDATES];
    xmedia_u32 candidate_count = 0;
    xmedia_u32 raw_candidate_count = 0;
    xmedia_s32 level_count;
    xmedia_s32 level_index;

    memset(levels, 0, sizeof(levels));
    level_count = collect_detect_levels(output, levels, 3);

    printf("\n========================================\n");
    printf("Candidate Boxes\n");
    printf("========================================\n");

    if (level_count <= 0) {
        printf("No valid detection levels found.\n");
        return;
    }

    for (level_index = 0; level_index < level_count; level_index++) {
        detect_level_info *level = &levels[level_index];
        xmedia_cl_tensor *reg_tensor;
        xmedia_cl_tensor *cls_tensor;
        xmedia_bool y_mapping_notice_printed = XMEDIA_FALSE;
        xmedia_u32 y;
        xmedia_u32 x;

        if (level->reg_index < 0 || level->cls_index < 0) {
            printf("Skip level %d: incomplete tensor pair for %ux%u\n",
                level_index, level->width, level->height);
            continue;
        }

        reg_tensor = &output->tensor[level->reg_index];
        cls_tensor = &output->tensor[level->cls_index];

        printf("Level %d: stride=%u, reg_tensor=%d, cls_tensor=%d, reg_type=%d(scale=%.6f zp=%d), cls_type=%d(scale=%.6f zp=%d)\n",
            level_index, level->stride, level->reg_index, level->cls_index,
            reg_tensor->shape.type, reg_tensor->quant.scale, reg_tensor->quant.zp,
            cls_tensor->shape.type, cls_tensor->quant.scale, cls_tensor->quant.zp);
        printf("  reg dims=[%u,%u,%u,%u] size=%u, cls dims=[%u,%u,%u,%u] size=%u\n",
            reg_tensor->shape.dims[0], reg_tensor->shape.dims[1], reg_tensor->shape.dims[2], reg_tensor->shape.dims[3], reg_tensor->size,
            cls_tensor->shape.dims[0], cls_tensor->shape.dims[1], cls_tensor->shape.dims[2], cls_tensor->shape.dims[3], cls_tensor->size);
        printf("  derived grid=(%u x %u), stride=%u, width*stride=%u, height*stride=%u\n",
            level->width, level->height, level->stride,
            level->width * level->stride, level->height * level->stride);
        print_level_sample_points(reg_tensor, cls_tensor, level, (xmedia_u32)level_index);

        for (y = 0; y < level->height; y++) {
            for (x = 0; x < level->width; x++) {
                xmedia_float best_logit = tensor_value_to_float(cls_tensor, 0, y, x);
                xmedia_s32 best_class = 0;
                xmedia_float score;
                xmedia_float left;
                xmedia_float top;
                xmedia_float right;
                xmedia_float bottom;
                xmedia_float center_x;
                xmedia_float center_y;
                xmedia_float scale_x;
                xmedia_float scale_y;
                xmedia_float x1;
                xmedia_float y1;
                xmedia_float x2;
                xmedia_float y2;
                xmedia_u32 c;

                /* 找出当前 grid 点分数最高的类别。 */
                for (c = 1; c < YOLO_CLASS_NUM; c++) {
                    xmedia_float logit = tensor_value_to_float(cls_tensor, c, y, x);
                    if (logit > best_logit) {
                        best_logit = logit;
                        best_class = (xmedia_s32)c;
                    }
                }

                score = sigmoidf_safe(best_logit);
                if (best_class != PERSON_CLASS_ID) {
                    continue;
                }

                if (score < CANDIDATE_SCORE_THRESH) {
                    continue;
                }

                /* 解码四条边相对中心点的距离。 */
                left = decode_dfl_distance(reg_tensor, 0, y, x, level->stride);
                top = decode_dfl_distance(reg_tensor, 1, y, x, level->stride);
                right = decode_dfl_distance(reg_tensor, 2, y, x, level->stride);
                bottom = decode_dfl_distance(reg_tensor, 3, y, x, level->stride);

                /* grid 中心点映射回输入坐标系。 */
                center_x = ((xmedia_float)x + 0.5f) * level->stride;
                center_y = ((xmedia_float)y + 0.5f) * level->stride;
                scale_x = level->width == 0 ? 1.0f : ((xmedia_float)IMAGE_WIDTH / (level->width * level->stride));

                /*
                 * 当前模型的检测头高度对应 384（48*8 / 24*16 / 12*32），
                 * 但输入图像高度是 360。实测直接乘 360/384 会让框整体上移。
                 * 这里先按“顶部对齐、底部补齐”的方式处理：Y 不再缩放，
                 * 只在最终 clamp 到 0~359 范围内。
                 */
                scale_y = 1.0f;

                if ((level->height * level->stride != IMAGE_HEIGHT) &&
                    (y_mapping_notice_printed == XMEDIA_FALSE)) {
                    printf("  notice: level %d grid height maps to %u, keep Y unscaled for top-aligned padded rows\n",
                        level_index, level->height * level->stride);
                    y_mapping_notice_printed = XMEDIA_TRUE;
                }

                x1 = clampf_safe((center_x - left) * scale_x, 0.0f, (xmedia_float)(IMAGE_WIDTH - 1));
                y1 = clampf_safe((center_y - top) * scale_y, 0.0f, (xmedia_float)(IMAGE_HEIGHT - 1));
                x2 = clampf_safe((center_x + right) * scale_x, 0.0f, (xmedia_float)(IMAGE_WIDTH - 1));
                y2 = clampf_safe((center_y + bottom) * scale_y, 0.0f, (xmedia_float)(IMAGE_HEIGHT - 1));

                if (x2 <= x1 || y2 <= y1) {
                    continue;
                }

                if (candidate_count < MAX_CANDIDATES) {
                    if (candidate_count < DEBUG_SAMPLE_LOCATIONS) {
                        printf("  raw_candidate[%u] grid=(%u,%u) center=(%.2f,%.2f) ltrb=(%.2f,%.2f,%.2f,%.2f) scale=(%.4f,%.4f) box=(%.2f,%.2f,%.2f,%.2f) score=%.4f class=%d\n",
                            candidate_count, x, y, center_x, center_y, left, top, right, bottom,
                            scale_x, scale_y, x1, y1, x2, y2, score, best_class);
                    }
                    candidates[candidate_count].class_id = best_class;
                    candidates[candidate_count].score = score;
                    candidates[candidate_count].x1 = x1;
                    candidates[candidate_count].y1 = y1;
                    candidates[candidate_count].x2 = x2;
                    candidates[candidate_count].y2 = y2;
                    candidates[candidate_count].stride = level->stride;
                    candidates[candidate_count].level = (xmedia_u32)level_index;
                    candidate_count++;
                }
            }
        }
    }

    raw_candidate_count = candidate_count;

    if (candidate_count == 0) {
        printf("No candidate boxes above threshold %.2f\n", CANDIDATE_SCORE_THRESH);
        return;
    }

    /* 先按分数排序，再做 NMS，符合常规目标检测后处理流程。 */
    qsort(candidates, candidate_count, sizeof(detect_candidate), compare_candidate_desc);
    candidate_count = apply_nms(candidates, candidate_count, NMS_IOU_THRESH);

    printf("Person class only (class_id=%d)\n", PERSON_CLASS_ID);
    printf("Found %u raw candidate boxes above threshold %.2f\n", raw_candidate_count, CANDIDATE_SCORE_THRESH);
    printf("After class-wise NMS (IoU <= %.2f): %u boxes\n", NMS_IOU_THRESH, candidate_count);
    if (candidate_count > PRINT_TOPK_CANDIDATES) {
        printf("Showing top %d candidates only\n", PRINT_TOPK_CANDIDATES);
        candidate_count = PRINT_TOPK_CANDIDATES;
    }

    for (level_index = 0; level_index < (xmedia_s32)candidate_count; level_index++) {
        printf("[%03d] cls=%d score=%.4f box=(%.1f, %.1f, %.1f, %.1f) stride=%u level=%u\n",
            level_index,
            candidates[level_index].class_id,
            candidates[level_index].score,
            candidates[level_index].x1,
            candidates[level_index].y1,
            candidates[level_index].x2,
            candidates[level_index].y2,
            candidates[level_index].stride,
            candidates[level_index].level);
    }
}

/*
 * MMZ内存分配 (完全仿照xmm.c)
 */
static xmedia_s32 mmz_alloc(xmedia_u64 *phy_addr, void **vir_addr, xmedia_u32 size)
{
    *phy_addr = xmedia_mmz_alloc(NULL, "tst", size);
    if (*phy_addr == 0) {
        return -1;
    }

    *vir_addr = xmedia_mmz_map(*phy_addr, size, 0);
    if (*vir_addr == NULL) {
        xmedia_mmz_free(*phy_addr);
        *phy_addr = 0;
        return -1;
    }

    return 0;
}

/*
 * MMZ内存释放 (完全仿照xmm.c)
 */
static xmedia_s32 mmz_free(xmedia_u64 phy_addr, void *vir_addr)
{
    if (vir_addr) {
        xmedia_mmz_unmap(vir_addr);
    }
    if (phy_addr) {
        xmedia_mmz_free(phy_addr);
    }
    return 0;
}

/*
 * 打印tensor输出信息
 */
/*
 * 主函数 - 完全仿照xmm.c格式
 */
xmedia_s32 main(xmedia_s32 argc, xmedia_char *argv[])
{
    xmedia_s32 ret;
    xmedia_s32 err;
    xmedia_s64 image_file_size = 0;
    
    /* 模型路径 */
    const char *model_path = MODEL_PATH;
    
    if (argc < 2) {
        printf("Usage: %s <image_path>\n", argv[0]);
        printf("Example: %s ./test.jpg\n", argv[0]);
        return -1;
    }
    
    const char *image_path = argv[1];
    
    printf("=== NPU Image Inference Test (xmm mode) ===\n");
    printf("Image: %s (%dx%d)\n", image_path, IMAGE_WIDTH, IMAGE_HEIGHT);
    printf("Model: %s\n", model_path);
    image_file_size = get_file_size(image_path);
    if (image_file_size >= 0) {
        printf("Image file size: %lld bytes\n", (long long)image_file_size);
    } else {
        printf("Warning: stat image file failed: %s\n", image_path);
    }
    if (!is_jpeg_file(image_path)) {
        printf("Error: only .jpg/.jpeg input is supported now\n");
        return -1;
    }

    /*
     * 以下变量基本对应 xmm 示例中的运行资源：
     * context / graph        : CL 运行时上下文与模型图
     * input / output         : 输入输出 tensor 元信息
     * *_phy_addr / *_vir_addr: MMZ 物理地址和虚拟地址
     */
    /* xmm相关变量 - 完全仿照xmm.c */
    xmedia_cl_context context = NULL;
    xmedia_cl_device_id *devices = NULL;
    xmedia_cl_u32 num_devices = 0;
    xmedia_cl_graph graph = NULL;
    xmedia_cl_tensor_info_inout input = {0};
    xmedia_cl_tensor_info_inout output = {0};
    
    /* 内存地址 */
    xmedia_u64 work_phy_addr = 0, weight_phy_addr = 0;
    xmedia_u64 input_phy_addr = 0, output_phy_addr = 0;
    void *work_vir_addr = NULL, *weight_vir_addr = NULL;
    void *input_vir_addr = NULL, *output_vir_addr = NULL;
    
    /* 内存大小 */
    xmedia_cl_u32 worksize = 0, weightsize = 0;
    xmedia_u32 inputsize = 0, outputsize = 0;
    xmedia_cl_s32 err_code = 0;
    
    /* 输入输出tensor内存 */
    xmedia_cl_tensor *input_tensor_arr = NULL;
    xmedia_cl_tensor *output_tensor_arr = NULL;
    
    /* 图片数据缓冲区 */
    xmedia_u8 *img_buffer = NULL;

    /* ==================== 步骤1: 初始化系统 ==================== */
    printf("\n[Step 1] Initialize system...\n");
    ret = xmedia_sys_init(NULL);
    if (ret != 0) {
        printf("xmedia_sys_init err, errno %d\n", ret);
        return ret;
    }
    printf("System initialized.\n");

    /* ==================== 步骤2: 初始化CL运行时 ==================== */
    printf("\n[Step 2] Initialize CL runtime...\n");
    ret = xmedia_cl_init();
    if (ret != 0) {
        printf("xmedia_cl_init err, errno %d\n", ret);
        goto SYS_EXIT;
    }
    printf("CL runtime initialized.\n");

    /* ==================== 步骤3: 获取设备 (第一次调用) ==================== */
    printf("\n[Step 3] Get CL devices (1st call)...\n");
    ret = xmedia_cl_get_device_ids(XMEDIA_CL_DEVICE_ALL, NULL, &num_devices);
    if (ret != 0) {
        printf("xmedia_cl_get_device_ids err, errno %d\n", ret);
        goto CL_EXIT;
    }
    printf("Found %d CL device(s)\n", num_devices);

    /* ==================== 步骤4: 获取设备 (第二次调用) ==================== */
    printf("\n[Step 4] Get CL devices (2nd call)...\n");
    devices = (xmedia_cl_device_id *)calloc(num_devices, sizeof(xmedia_cl_device_id));
    if (devices == NULL) {
        printf("calloc devices failed\n");
        ret = -1;
        goto CL_EXIT;
    }

    ret = xmedia_cl_get_device_ids(XMEDIA_CL_DEVICE_ALL, devices, &num_devices);
    if (ret != 0) {
        printf("xmedia_cl_get_device_ids err, errno %d\n", ret);
        free(devices);
        goto CL_EXIT;
    }
    printf("Devices acquired.\n");

    /* ==================== 步骤5: 创建上下文 ==================== */
    printf("\n[Step 5] Create CL context...\n");
    context = xmedia_cl_create_context(num_devices, devices, &err_code);
    if (err_code != 0) {
        printf("xmedia_cl_create_context err, errno %d, err_code %d\n", ret, err_code);
        free(devices);
        goto CL_EXIT;
    }
    printf("Context created: %p\n", context);

    /* ==================== 步骤6: 查询模型大小 ==================== */
    printf("\n[Step 6] Query model size...\n");
    ret = xmedia_cl_graph_querysize_from_file(model_path, &worksize, &weightsize);
    if (ret != 0) {
        printf("xmedia_cl_graph_querysize_from_file err, errno %d\n", ret);
        goto FREE_CONTEXT;
    }
    printf("Model work size: %u bytes, weight size: %u bytes\n", worksize, weightsize);

    /* ==================== 步骤7: 分配work内存 ==================== */
    printf("\n[Step 7] Allocate work memory...\n");
    if (worksize > 0) {
        ret = mmz_alloc(&work_phy_addr, &work_vir_addr, worksize);
        if (ret != 0) {
            printf("allocate work memory failed\n");
            goto FREE_CONTEXT;
        }
    } else {
        work_vir_addr = NULL;
    }
    printf("Work memory allocated: phy=0x%llx, vir=%p\n", 
           (unsigned long long)work_phy_addr, work_vir_addr);

    /* ==================== 步骤8: 分配weight内存 ==================== */
    printf("\n[Step 8] Allocate weight memory...\n");
    if (weightsize > 0) {
        ret = mmz_alloc(&weight_phy_addr, &weight_vir_addr, weightsize);
        if (ret != 0) {
            printf("allocate weight memory failed\n");
            goto FREE_WORK;
        }
    } else {
        weight_vir_addr = NULL;
    }
    printf("Weight memory allocated: phy=0x%llx, vir=%p\n", 
           (unsigned long long)weight_phy_addr, weight_vir_addr);

    /* ==================== 步骤9: 加载模型 ==================== */
    printf("\n[Step 9] Load model from file...\n");
    ret = xmedia_cl_graph_loadmodel_from_file_withmem(
        &context, 
        model_path, 
        work_vir_addr, worksize,
        weight_vir_addr, weightsize,
        &graph);
    if (ret != 0) {
        printf("xmedia_cl_graph_loadmodel_from_file_withmem err, errno %d\n", ret);
        goto FREE_WEIGHT;
    }
    printf("Model loaded, graph: %p\n", graph);

    /* ==================== 步骤10: 获取输入信息 (第一次调用) ==================== */
    printf("\n[Step 10] Get input tensor info (1st call)...\n");
    ret = xmedia_cl_graph_get_input(graph, 0, &input);
    if (ret != 0) {
        printf("xmedia_cl_graph_get_input err, errno %d\n", ret);
        goto UNLOAD_GRAPH;
    }
    printf("Input tensor count: %d\n", input.num);

    /* ==================== 步骤11: 获取输入信息 (第二次调用) ==================== */
    printf("\n[Step 11] Get input tensor info (2nd call)...\n");
    /* 分配input tensor数组 */
    input_tensor_arr = (xmedia_cl_tensor *)calloc(input.num, sizeof(xmedia_cl_tensor));
    if (input_tensor_arr == NULL) {
        printf("calloc input tensor failed\n");
        ret = -1;
        goto UNLOAD_GRAPH;
    }
    input.tensor = input_tensor_arr;
    
    ret = xmedia_cl_graph_get_input(graph, input.num, &input);
    if (ret != 0) {
        printf("xmedia_cl_graph_get_input err, errno %d\n", ret);
        goto FREE_INPUT_TENSOR;
    }
    print_tensor_layout("Input Tensor Layout", &input);

    /* ==================== 步骤12: 获取输出信息 (第一次调用) ==================== */
    printf("\n[Step 12] Get output tensor info (1st call)...\n");
    ret = xmedia_cl_graph_get_output(graph, 0, &output);
    if (ret != 0) {
        printf("xmedia_cl_graph_get_output err, errno %d\n", ret);
        goto FREE_INPUT_TENSOR;
    }
    printf("Output tensor count: %d\n", output.num);

    /* ==================== 步骤13: 获取输出信息 (第二次调用) ==================== */
    printf("\n[Step 13] Get output tensor info (2nd call)...\n");
    /* 分配output tensor数组 */
    output_tensor_arr = (xmedia_cl_tensor *)calloc(output.num, sizeof(xmedia_cl_tensor));
    if (output_tensor_arr == NULL) {
        printf("calloc output tensor failed\n");
        ret = -1;
        goto FREE_INPUT_TENSOR;
    }
    output.tensor = output_tensor_arr;
    
    ret = xmedia_cl_graph_get_output(graph, output.num, &output);
    if (ret != 0) {
        printf("xmedia_cl_graph_get_output err, errno %d\n", ret);
        goto FREE_OUTPUT_TENSOR;
    }
    printf("Output tensor metadata ready.\n");
    print_tensor_layout("Output Tensor Layout", &output);

    /*
     * 输入/输出 tensor 的 size 是每个 tensor 的有效大小，
     * 真正分配总内存时按 ALIGN_BYTE 对齐后顺序拼接，和 xmm 示例保持一致。
     */
    /* ==================== 步骤14: 计算输入大小并分配内存 ==================== */
    printf("\n[Step 14] Allocate input memory...\n");
    inputsize = 0;
    for (xmedia_s32 i = 0; i < (xmedia_s32)input.num; i++) {
        inputsize += ALIGN_FUNC(input.tensor[i].size, ALIGN_BYTE);
        printf("Input[%d]: size=%u bytes\n", i, input.tensor[i].size);
    }
    printf("Total input size: %u bytes\n", inputsize);

    ret = mmz_alloc(&input_phy_addr, &input_vir_addr, inputsize);
    if (ret != 0) {
        printf("allocate input memory failed\n");
        goto FREE_OUTPUT_TENSOR;
    }
    printf("Input memory allocated: phy=0x%llx, vir=%p\n", 
           (unsigned long long)input_phy_addr, input_vir_addr);

    /* ==================== 步骤15: 计算输出大小并分配内存 ==================== */
    printf("\n[Step 15] Allocate output memory...\n");
    outputsize = 0;
    for (xmedia_s32 i = 0; i < (xmedia_s32)output.num; i++) {
        outputsize += ALIGN_FUNC(output.tensor[i].size, ALIGN_BYTE);
        printf("Output[%d]: size=%u bytes\n", i, output.tensor[i].size);
    }
    printf("Total output size: %u bytes\n", outputsize);

    ret = mmz_alloc(&output_phy_addr, &output_vir_addr, outputsize);
    if (ret != 0) {
        printf("allocate output memory failed\n");
        goto FREE_INPUT;
    }
    printf("Output memory allocated: phy=0x%llx, vir=%p\n", 
           (unsigned long long)output_phy_addr, output_vir_addr);

    /*
     * 将大块 MMZ 内存切分给每个 input tensor。
     * 每个 tensor 的起始地址都基于前一个 tensor 的“对齐后大小”顺延。
     */
    /* ==================== 步骤16: 设置输入tensor地址 ==================== */
    printf("\n[Step 16] Setup input tensor addresses...\n");
    xmedia_u32 offset = 0;
    for (xmedia_s32 i = 0; i < (xmedia_s32)input.num; i++) {
        if (i > 0) {
            input.tensor[i].addr = (xmedia_u8 *)input.tensor[i-1].addr + ALIGN_FUNC(input.tensor[i-1].size, ALIGN_BYTE);
        } else {
            input.tensor[i].addr = input_vir_addr;
        }
        offset += ALIGN_FUNC(input.tensor[i].size, ALIGN_BYTE);
        printf("Input[%d] addr: %p, size: %u\n", i, input.tensor[i].addr, input.tensor[i].size);
    }

    /* 输出 tensor 的地址组织方式与 input 相同。 */
    /* ==================== 步骤17: 设置输出tensor地址 ==================== */
    printf("\n[Step 17] Setup output tensor addresses...\n");
    offset = 0;
    for (xmedia_s32 i = 0; i < (xmedia_s32)output.num; i++) {
        if (i > 0) {
            output.tensor[i].addr = (xmedia_u8 *)output.tensor[i-1].addr + ALIGN_FUNC(output.tensor[i-1].size, ALIGN_BYTE);
        } else {
            output.tensor[i].addr = output_vir_addr;
        }
        offset += ALIGN_FUNC(output.tensor[i].size, ALIGN_BYTE);
        printf("Output[%d] addr: %p, size: %u\n", i, output.tensor[i].addr, output.tensor[i].size);
    }

    /* ==================== 步骤18: 读取JPG并转RGB ==================== */
    printf("\n[Step 18] Decode JPG to RGB24...\n");
    xmedia_u32 rgb_size = IMAGE_WIDTH * IMAGE_HEIGHT * 3;
    xmedia_u32 channel_size = IMAGE_WIDTH * IMAGE_HEIGHT;
    img_buffer = NULL;

    ret = decode_jpeg_to_rgb24(image_path, &img_buffer, IMAGE_WIDTH, IMAGE_HEIGHT);
    if (ret != 0 || img_buffer == NULL) {
        printf("decode_jpeg_to_rgb24 failed\n");
        goto FREE_OUTPUT;
    }
    printf("RGB image ready: %u bytes\n", rgb_size);
    dump_bytes_hex("RGB pixel stream first bytes", img_buffer, rgb_size);

    /*
     * 当前模型只有一个输入，且数据格式为 8bit CHW RGB。
     * 这里先清零，再把 RGB24 数据转换并拷贝进去，避免残留脏数据影响推理结果。
     */
    /* ==================== 步骤19: 复制RGB数据到输入tensor(CHW) ==================== */
    printf("\n[Step 19] Convert RGB24 to model input tensor...\n");
    if (input.num != 1) {
        printf("Warning: input tensor count is %d, current logic only fills tensor[0]\n", input.num);
    }
    printf("Input copy debug: channel_size=%u, required_chw_bytes=%u, input_tensor0_size=%u\n",
        channel_size, channel_size * 3, input.tensor[0].size);
    if (channel_size * 3 > input.tensor[0].size) {
        printf("Error: input tensor[0] size %u is smaller than RGB CHW size %u\n",
            input.tensor[0].size, channel_size * 3);
        ret = -1;
        goto FREE_IMG;
    }
    memset(input.tensor[0].addr, 0, input.tensor[0].size);
    rgb24_to_chw(img_buffer, (xmedia_u8 *)input.tensor[0].addr, IMAGE_WIDTH, IMAGE_HEIGHT);
    printf("RGB24 converted to CHW tensor.\n");
    dump_bytes_hex("Input tensor channel R first bytes", (xmedia_u8 *)input.tensor[0].addr, channel_size);
    dump_bytes_hex("Input tensor channel G first bytes", (xmedia_u8 *)input.tensor[0].addr + channel_size, channel_size);
    dump_bytes_hex("Input tensor channel B first bytes", (xmedia_u8 *)input.tensor[0].addr + 2 * channel_size, channel_size);

    /* ==================== 步骤20: 设置输入输出到graph ==================== */
    printf("\n[Step 20] Set input/output to graph...\n");
    ret = xmedia_cl_graph_set_inout(graph, &input, &output);
    if (ret != 0) {
        printf("xmedia_cl_graph_set_inout err, errno %d\n", ret);
        goto FREE_IMG;
    }
    printf("Input/Output set to graph.\n");

    /* ==================== 步骤21: 执行推理 ==================== */
    printf("\n[Step 21] Execute inference...\n");
    struct timeval t_start, t_end;
    gettimeofday(&t_start, NULL);
    
    ret = xmedia_cl_graph_process(graph);
    
    gettimeofday(&t_end, NULL);
    xmedia_s64 elapsed = (t_end.tv_sec - t_start.tv_sec) * 1000000 + (t_end.tv_usec - t_start.tv_usec);
    
    if (ret != 0) {
        printf("xmedia_cl_graph_process err, errno %d\n", ret);
    } else {
        printf("Inference completed successfully!\n");
    }
    printf("Inference time: %lld us (%lld ms)\n", elapsed, elapsed / 1000);

    /* ==================== 步骤22: 解析候选框 ==================== */
    print_candidate_boxes(&output);

    /*
     * 资源释放顺序与申请顺序相反，保持 goto 错误退出路径也能安全回收资源。
     */
    /* ==================== 释放资源 ==================== */
    printf("\n[Free Resources]...\n");
FREE_IMG:
    if (img_buffer) {
        free(img_buffer);
        img_buffer = NULL;
    }
FREE_OUTPUT:
    mmz_free(output_phy_addr, output_vir_addr);
FREE_INPUT:
    mmz_free(input_phy_addr, input_vir_addr);
FREE_OUTPUT_TENSOR:
    if (output_tensor_arr) {
        free(output_tensor_arr);
        output_tensor_arr = NULL;
    }
FREE_INPUT_TENSOR:
    if (input_tensor_arr) {
        free(input_tensor_arr);
        input_tensor_arr = NULL;
    }
UNLOAD_GRAPH:
    if (graph) {
        err = xmedia_cl_graph_unload(graph);
        if (err != 0) {
            printf("xmedia_cl_graph_unload err, errno %d\n", err);
        }
    }
FREE_WEIGHT:
    mmz_free(weight_phy_addr, weight_vir_addr);
FREE_WORK:
    mmz_free(work_phy_addr, work_vir_addr);
FREE_CONTEXT:
    if (context) {
        err = xmedia_cl_release_context(context);
        if (err != 0) {
            printf("xmedia_cl_release_context err, errno %d\n", err);
        }
    }
    if (devices) {
        err = xmedia_cl_release_device_ids(devices, &num_devices);
        if (err != 0) {
            printf("xmedia_cl_release_device_ids err, errno %d\n", err);
        }
        free(devices);
    }
CL_EXIT:
    err = xmedia_cl_uninit();
    if (err != 0) {
        printf("xmedia_cl_uninit err, errno %d\n", err);
    }
SYS_EXIT:
    err = xmedia_sys_exit();
    if (err != 0) {
        printf("xmedia_sys_exit err, errno %d\n", err);
    }

    printf("\n=== Test Completed ===\n");

    return ret;
}
