#include "../include/xm_image_infer.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "xmedia_mmz.h"
#include "xmedia_sys.h"

#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/imgutils.h"
#include "libavutil/pixfmt.h"
#include "libswscale/swscale.h"

#define XM_ALIGN_BYTE 16
#define XM_DFL_BINS 16
#define XM_YOLO_CLASS_NUM 80
#define XM_MAX_CANDIDATES 6000

#define XM_ALIGN_FUNC(A, ALIGN) ((((A) % (ALIGN)) == 0) ? (A) : ((A) + (ALIGN) - ((A) % (ALIGN))))

typedef struct {
    xmedia_s32 reg_index;
    xmedia_s32 cls_index;
    xmedia_u32 height;
    xmedia_u32 width;
    xmedia_u32 stride;
} detect_level_info;

struct xm_image_infer_handle {
    xm_image_infer_config config;
    xmedia_cl_context context;
    xmedia_cl_device_id *devices;
    xmedia_cl_u32 num_devices;
    xmedia_cl_graph graph;
    xmedia_cl_tensor_info_inout input;
    xmedia_cl_tensor_info_inout output;
    xmedia_cl_tensor *input_tensor_arr;
    xmedia_cl_tensor *output_tensor_arr;
    xmedia_u64 work_phy_addr;
    xmedia_u64 weight_phy_addr;
    xmedia_u64 input_phy_addr;
    xmedia_u64 output_phy_addr;
    void *work_vir_addr;
    void *weight_vir_addr;
    void *input_vir_addr;
    void *output_vir_addr;
    xmedia_cl_u32 worksize;
    xmedia_cl_u32 weightsize;
    xmedia_u32 inputsize;
    xmedia_u32 outputsize;
    xmedia_bool sys_inited;
    xmedia_bool cl_inited;
};

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

static xmedia_s32 mmz_alloc(xmedia_u64 *phy_addr, void **vir_addr, xmedia_u32 size)
{
    *phy_addr = xmedia_mmz_alloc(NULL, "xmimg", size);
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

static xmedia_void mmz_release(xmedia_u64 phy_addr, void *vir_addr)
{
    if (vir_addr != NULL) {
        xmedia_mmz_unmap(vir_addr);
    }
    if (phy_addr != 0) {
        xmedia_mmz_free(phy_addr);
    }
}

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
        goto EXIT;
    }

    if (avformat_find_stream_info(format_context, NULL) < 0) {
        goto EXIT;
    }

    stream_index = av_find_best_stream(format_context, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
    if (stream_index < 0 || codec == NULL) {
        goto EXIT;
    }

    codec_context = avcodec_alloc_context3(codec);
    if (codec_context == NULL) {
        goto EXIT;
    }

    if (avcodec_parameters_to_context(codec_context,
        format_context->streams[stream_index]->codecpar) < 0) {
        goto EXIT;
    }

    if (avcodec_open2(codec_context, codec, NULL) < 0) {
        goto EXIT;
    }

    packet = av_packet_alloc();
    frame = av_frame_alloc();
    frame_rgb = av_frame_alloc();
    if (packet == NULL || frame == NULL || frame_rgb == NULL) {
        goto EXIT;
    }

    num_bytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24, dst_width, dst_height, 1);
    if (num_bytes <= 0) {
        goto EXIT;
    }

    buffer = (xmedia_u8 *)malloc(num_bytes);
    if (buffer == NULL) {
        goto EXIT;
    }

    if (av_image_fill_arrays(frame_rgb->data, frame_rgb->linesize, buffer,
        AV_PIX_FMT_RGB24, dst_width, dst_height, 1) < 0) {
        goto EXIT;
    }

    sws_context = sws_getContext(codec_context->width, codec_context->height, codec_context->pix_fmt,
        dst_width, dst_height, AV_PIX_FMT_RGB24, SWS_BILINEAR, NULL, NULL, NULL);
    if (sws_context == NULL) {
        goto EXIT;
    }

    while (av_read_frame(format_context, packet) >= 0) {
        if (packet->stream_index != stream_index) {
            av_packet_unref(packet);
            continue;
        }

        if (avcodec_send_packet(codec_context, packet) < 0) {
            av_packet_unref(packet);
            goto EXIT;
        }
        av_packet_unref(packet);

        while (avcodec_receive_frame(codec_context, frame) >= 0) {
            sws_scale(sws_context, (const xmedia_u8 * const *)frame->data, frame->linesize,
                0, frame->height, frame_rgb->data, frame_rgb->linesize);
            *rgb_buffer = buffer;
            buffer = NULL;
            ret = 0;
            goto EXIT;
        }
    }

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

static xmedia_u32 get_tensor_bytes_per_element(const xmedia_cl_tensor *tensor)
{
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

    if (channels == 0 || height == 0 || bytes_per_elem == 0 || tensor->size == 0) {
        return logical_width;
    }

    return tensor->size / (channels * height * bytes_per_elem);
}

static xmedia_float tensor_value_to_float(const xmedia_cl_tensor *tensor, xmedia_u32 channel,
    xmedia_u32 y, xmedia_u32 x)
{
    xmedia_u32 height = tensor->shape.dims[2];
    xmedia_u32 width = get_tensor_physical_width(tensor);
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

static xmedia_float sigmoidf_safe(xmedia_float x)
{
    if (x >= 0.0f) {
        xmedia_float z = expf(-x);
        return 1.0f / (1.0f + z);
    }

    xmedia_float z = expf(x);
    return z / (1.0f + z);
}

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

static xmedia_float decode_dfl_distance(const xmedia_cl_tensor *reg_tensor,
    xmedia_u32 side_index, xmedia_u32 y, xmedia_u32 x, xmedia_u32 stride)
{
    xmedia_float logits[XM_DFL_BINS];
    xmedia_float max_logit;
    xmedia_float sum = 0.0f;
    xmedia_float expectation = 0.0f;
    xmedia_u32 k;
    xmedia_u32 base_channel = side_index * XM_DFL_BINS;

    max_logit = tensor_value_to_float(reg_tensor, base_channel, y, x);
    for (k = 0; k < XM_DFL_BINS; k++) {
        logits[k] = tensor_value_to_float(reg_tensor, base_channel + k, y, x);
        if (logits[k] > max_logit) {
            max_logit = logits[k];
        }
    }

    for (k = 0; k < XM_DFL_BINS; k++) {
        logits[k] = expf(logits[k] - max_logit);
        sum += logits[k];
    }

    if (sum <= 0.0f) {
        return 0.0f;
    }

    for (k = 0; k < XM_DFL_BINS; k++) {
        expectation += ((xmedia_float)k) * (logits[k] / sum);
    }

    return expectation * stride;
}

static xmedia_s32 compare_candidate_desc(const void *left, const void *right)
{
    const xm_detect_box *a = (const xm_detect_box *)left;
    const xm_detect_box *b = (const xm_detect_box *)right;

    if (a->score < b->score) {
        return 1;
    }
    if (a->score > b->score) {
        return -1;
    }
    return 0;
}

static xmedia_float compute_iou(const xm_detect_box *a, const xm_detect_box *b)
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

static xmedia_u32 apply_nms(xm_detect_box *candidates, xmedia_u32 candidate_count,
    xmedia_float iou_thresh)
{
    xmedia_u8 *suppressed;
    xmedia_u32 keep_count = 0;
    xmedia_u32 i;
    xmedia_u32 j;

    suppressed = (xmedia_u8 *)calloc(candidate_count, sizeof(xmedia_u8));
    if (suppressed == NULL) {
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

static xmedia_s32 collect_detect_levels(const xm_image_infer_handle *handle,
    detect_level_info *levels, xmedia_u32 max_levels)
{
    xmedia_u32 i;
    xmedia_u32 level_count = 0;
    xmedia_cl_tensor_info_inout *output = (xmedia_cl_tensor_info_inout *)&handle->output;

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
            levels[level_count].stride = width == 0 ? 0 : (handle->config.image_width / width);
            level_count++;
        }

        if (channels == XM_DFL_BINS * 4) {
            levels[matched].reg_index = (xmedia_s32)i;
        } else if (channels == XM_YOLO_CLASS_NUM) {
            levels[matched].cls_index = (xmedia_s32)i;
        }
    }

    return (xmedia_s32)level_count;
}

static xmedia_s32 collect_candidate_boxes(const xm_image_infer_handle *handle,
    xm_detect_result *result)
{
    detect_level_info levels[3];
    xm_detect_box candidates[XM_MAX_CANDIDATES];
    xmedia_u32 candidate_count = 0;
    xmedia_s32 level_count;
    xmedia_s32 level_index;

    memset(levels, 0, sizeof(levels));
    level_count = collect_detect_levels(handle, levels, 3);
    if (level_count <= 0) {
        result->boxes = NULL;
        result->count = 0;
        return 0;
    }

    for (level_index = 0; level_index < level_count; level_index++) {
        detect_level_info *level = &levels[level_index];
        xmedia_cl_tensor *reg_tensor;
        xmedia_cl_tensor *cls_tensor;
        xmedia_u32 y;
        xmedia_u32 x;

        if (level->reg_index < 0 || level->cls_index < 0) {
            continue;
        }

        reg_tensor = &handle->output.tensor[level->reg_index];
        cls_tensor = &handle->output.tensor[level->cls_index];

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

                for (c = 1; c < XM_YOLO_CLASS_NUM; c++) {
                    xmedia_float logit = tensor_value_to_float(cls_tensor, c, y, x);
                    if (logit > best_logit) {
                        best_logit = logit;
                        best_class = (xmedia_s32)c;
                    }
                }

                score = sigmoidf_safe(best_logit);
                if (best_class != (xmedia_s32)handle->config.person_class_id ||
                    score < handle->config.score_thresh) {
                    continue;
                }

                left = decode_dfl_distance(reg_tensor, 0, y, x, level->stride);
                top = decode_dfl_distance(reg_tensor, 1, y, x, level->stride);
                right = decode_dfl_distance(reg_tensor, 2, y, x, level->stride);
                bottom = decode_dfl_distance(reg_tensor, 3, y, x, level->stride);

                center_x = ((xmedia_float)x + 0.5f) * level->stride;
                center_y = ((xmedia_float)y + 0.5f) * level->stride;
                scale_x = level->width == 0 ? 1.0f :
                    ((xmedia_float)handle->config.image_width / (level->width * level->stride));
                scale_y = 1.0f;

                x1 = clampf_safe((center_x - left) * scale_x, 0.0f,
                    (xmedia_float)(handle->config.image_width - 1));
                y1 = clampf_safe((center_y - top) * scale_y, 0.0f,
                    (xmedia_float)(handle->config.image_height - 1));
                x2 = clampf_safe((center_x + right) * scale_x, 0.0f,
                    (xmedia_float)(handle->config.image_width - 1));
                y2 = clampf_safe((center_y + bottom) * scale_y, 0.0f,
                    (xmedia_float)(handle->config.image_height - 1));

                if ((x2 <= x1) || (y2 <= y1) || candidate_count >= XM_MAX_CANDIDATES) {
                    continue;
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

    if (candidate_count == 0) {
        result->boxes = NULL;
        result->count = 0;
        return 0;
    }

    qsort(candidates, candidate_count, sizeof(xm_detect_box), compare_candidate_desc);
    candidate_count = apply_nms(candidates, candidate_count, handle->config.nms_iou_thresh);

    result->boxes = (xm_detect_box *)calloc(candidate_count, sizeof(xm_detect_box));
    if (result->boxes == NULL) {
        result->count = 0;
        return -1;
    }

    memcpy(result->boxes, candidates, candidate_count * sizeof(xm_detect_box));
    result->count = candidate_count;
    return 0;
}

static xmedia_s32 assign_tensor_addrs(xmedia_cl_tensor_info_inout *tensor_info, void *base_addr)
{
    xmedia_s32 i;

    for (i = 0; i < (xmedia_s32)tensor_info->num; i++) {
        if (i > 0) {
            tensor_info->tensor[i].addr = (xmedia_u8 *)tensor_info->tensor[i - 1].addr +
                XM_ALIGN_FUNC(tensor_info->tensor[i - 1].size, XM_ALIGN_BYTE);
        } else {
            tensor_info->tensor[i].addr = base_addr;
        }
    }

    return 0;
}

static xmedia_void fill_default_config(xm_image_infer_config *config)
{
    if (config->image_width == 0) {
        config->image_width = 640;
    }
    if (config->image_height == 0) {
        config->image_height = 360;
    }
    if (config->score_thresh <= 0.0f) {
        config->score_thresh = 0.45f;
    }
    if (config->nms_iou_thresh <= 0.0f) {
        config->nms_iou_thresh = 0.45f;
    }
}

xmedia_s32 xm_image_infer_init(const xm_image_infer_config *config,
    xm_image_infer_handle **handle)
{
    xm_image_infer_handle *ctx;
    xmedia_cl_s32 err_code = 0;
    xmedia_s32 ret;
    xmedia_s32 i;

    if (config == NULL || handle == NULL || config->model_path == NULL) {
        return -1;
    }

    *handle = NULL;
    ctx = (xm_image_infer_handle *)calloc(1, sizeof(*ctx));
    if (ctx == NULL) {
        return -1;
    }

    ctx->config = *config;
    fill_default_config(&ctx->config);

    ret = xmedia_sys_init(NULL);
    if (ret != 0) {
        xm_image_infer_destroy(ctx);
        return ret;
    }
    ctx->sys_inited = XMEDIA_TRUE;

    ret = xmedia_cl_init();
    if (ret != 0) {
        xm_image_infer_destroy(ctx);
        return ret;
    }
    ctx->cl_inited = XMEDIA_TRUE;

    ret = xmedia_cl_get_device_ids(XMEDIA_CL_DEVICE_ALL, NULL, &ctx->num_devices);
    if (ret != 0 || ctx->num_devices == 0) {
        xm_image_infer_destroy(ctx);
        return ret != 0 ? ret : -1;
    }

    ctx->devices = (xmedia_cl_device_id *)calloc(ctx->num_devices, sizeof(xmedia_cl_device_id));
    if (ctx->devices == NULL) {
        xm_image_infer_destroy(ctx);
        return -1;
    }

    ret = xmedia_cl_get_device_ids(XMEDIA_CL_DEVICE_ALL, ctx->devices, &ctx->num_devices);
    if (ret != 0) {
        xm_image_infer_destroy(ctx);
        return ret;
    }

    ctx->context = xmedia_cl_create_context(ctx->num_devices, ctx->devices, &err_code);
    if (err_code != 0 || ctx->context == NULL) {
        xm_image_infer_destroy(ctx);
        return err_code != 0 ? err_code : -1;
    }

    ret = xmedia_cl_graph_querysize_from_file(ctx->config.model_path, &ctx->worksize, &ctx->weightsize);
    if (ret != 0) {
        xm_image_infer_destroy(ctx);
        return ret;
    }

    if (ctx->worksize > 0 && mmz_alloc(&ctx->work_phy_addr, &ctx->work_vir_addr, ctx->worksize) != 0) {
        xm_image_infer_destroy(ctx);
        return -1;
    }

    if (ctx->weightsize > 0 && mmz_alloc(&ctx->weight_phy_addr, &ctx->weight_vir_addr, ctx->weightsize) != 0) {
        xm_image_infer_destroy(ctx);
        return -1;
    }

    ret = xmedia_cl_graph_loadmodel_from_file_withmem(&ctx->context,
        ctx->config.model_path, ctx->work_vir_addr, ctx->worksize,
        ctx->weight_vir_addr, ctx->weightsize, &ctx->graph);
    if (ret != 0) {
        xm_image_infer_destroy(ctx);
        return ret;
    }

    ret = xmedia_cl_graph_get_input(ctx->graph, 0, &ctx->input);
    if (ret != 0) {
        xm_image_infer_destroy(ctx);
        return ret;
    }

    ctx->input_tensor_arr = (xmedia_cl_tensor *)calloc(ctx->input.num, sizeof(xmedia_cl_tensor));
    if (ctx->input_tensor_arr == NULL) {
        xm_image_infer_destroy(ctx);
        return -1;
    }
    ctx->input.tensor = ctx->input_tensor_arr;

    ret = xmedia_cl_graph_get_input(ctx->graph, ctx->input.num, &ctx->input);
    if (ret != 0) {
        xm_image_infer_destroy(ctx);
        return ret;
    }

    ret = xmedia_cl_graph_get_output(ctx->graph, 0, &ctx->output);
    if (ret != 0) {
        xm_image_infer_destroy(ctx);
        return ret;
    }

    ctx->output_tensor_arr = (xmedia_cl_tensor *)calloc(ctx->output.num, sizeof(xmedia_cl_tensor));
    if (ctx->output_tensor_arr == NULL) {
        xm_image_infer_destroy(ctx);
        return -1;
    }
    ctx->output.tensor = ctx->output_tensor_arr;

    ret = xmedia_cl_graph_get_output(ctx->graph, ctx->output.num, &ctx->output);
    if (ret != 0) {
        xm_image_infer_destroy(ctx);
        return ret;
    }

    for (i = 0; i < (xmedia_s32)ctx->input.num; i++) {
        ctx->inputsize += XM_ALIGN_FUNC(ctx->input.tensor[i].size, XM_ALIGN_BYTE);
    }
    for (i = 0; i < (xmedia_s32)ctx->output.num; i++) {
        ctx->outputsize += XM_ALIGN_FUNC(ctx->output.tensor[i].size, XM_ALIGN_BYTE);
    }

    if (ctx->inputsize > 0 && mmz_alloc(&ctx->input_phy_addr, &ctx->input_vir_addr, ctx->inputsize) != 0) {
        xm_image_infer_destroy(ctx);
        return -1;
    }

    if (ctx->outputsize > 0 && mmz_alloc(&ctx->output_phy_addr, &ctx->output_vir_addr, ctx->outputsize) != 0) {
        xm_image_infer_destroy(ctx);
        return -1;
    }

    assign_tensor_addrs(&ctx->input, ctx->input_vir_addr);
    assign_tensor_addrs(&ctx->output, ctx->output_vir_addr);

    *handle = ctx;
    return 0;
}

xmedia_s32 xm_image_infer_detect(xm_image_infer_handle *handle,
    const xmedia_char *image_path, xm_detect_result *result)
{
    xmedia_s32 ret;
    xmedia_u8 *img_buffer = NULL;
    xmedia_u32 channel_size;

    if (handle == NULL || image_path == NULL || result == NULL) {
        return -1;
    }

    result->boxes = NULL;
    result->count = 0;

    if (!is_jpeg_file(image_path)) {
        return -1;
    }

    ret = decode_jpeg_to_rgb24(image_path, &img_buffer,
        handle->config.image_width, handle->config.image_height);
    if (ret != 0 || img_buffer == NULL) {
        return -1;
    }

    if (handle->input.num == 0 || handle->input.tensor[0].addr == NULL) {
        free(img_buffer);
        return -1;
    }

    channel_size = handle->config.image_width * handle->config.image_height;
    if (channel_size * 3 > handle->input.tensor[0].size) {
        free(img_buffer);
        return -1;
    }

    memset(handle->input.tensor[0].addr, 0, handle->input.tensor[0].size);
    rgb24_to_chw(img_buffer, (xmedia_u8 *)handle->input.tensor[0].addr,
        handle->config.image_width, handle->config.image_height);
    free(img_buffer);

    ret = xmedia_cl_graph_set_inout(handle->graph, &handle->input, &handle->output);
    if (ret != 0) {
        return ret;
    }

    ret = xmedia_cl_graph_process(handle->graph);
    if (ret != 0) {
        return ret;
    }

    return collect_candidate_boxes(handle, result);
}

xmedia_void xm_image_infer_result_deinit(xm_detect_result *result)
{
    if (result == NULL) {
        return;
    }

    if (result->boxes != NULL) {
        free(result->boxes);
        result->boxes = NULL;
    }
    result->count = 0;
}

xmedia_void xm_image_infer_destroy(xm_image_infer_handle *handle)
{
    xmedia_s32 err;

    if (handle == NULL) {
        return;
    }

    mmz_release(handle->output_phy_addr, handle->output_vir_addr);
    mmz_release(handle->input_phy_addr, handle->input_vir_addr);

    if (handle->output_tensor_arr != NULL) {
        free(handle->output_tensor_arr);
    }
    if (handle->input_tensor_arr != NULL) {
        free(handle->input_tensor_arr);
    }

    if (handle->graph != NULL) {
        err = xmedia_cl_graph_unload(handle->graph);
        if (err != 0) {
            printf("xmedia_cl_graph_unload err, errno %d\n", err);
        }
    }

    mmz_release(handle->weight_phy_addr, handle->weight_vir_addr);
    mmz_release(handle->work_phy_addr, handle->work_vir_addr);

    if (handle->context != NULL) {
        err = xmedia_cl_release_context(handle->context);
        if (err != 0) {
            printf("xmedia_cl_release_context err, errno %d\n", err);
        }
    }

    if (handle->devices != NULL) {
        err = xmedia_cl_release_device_ids(handle->devices, &handle->num_devices);
        if (err != 0) {
            printf("xmedia_cl_release_device_ids err, errno %d\n", err);
        }
        free(handle->devices);
    }

    if (handle->cl_inited == XMEDIA_TRUE) {
        err = xmedia_cl_uninit();
        if (err != 0) {
            printf("xmedia_cl_uninit err, errno %d\n", err);
        }
    }

    if (handle->sys_inited == XMEDIA_TRUE) {
        err = xmedia_sys_exit();
        if (err != 0) {
            printf("xmedia_sys_exit err, errno %d\n", err);
        }
    }

    free(handle);
}
