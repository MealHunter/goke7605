#include "../include/XC_image_infer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "xmedia_mmz.h"
#include "xmedia_sys.h"

#define XC_ALIGN_BYTE 16
#define XC_DFL_BINS 16
#define XC_YOLO_CLASS_NUM 80
#define XC_MAX_CANDIDATES 6000

#define XC_ALIGN_FUNC(A, ALIGN) ((((A) % (ALIGN)) == 0) ? (A) : ((A) + (ALIGN) - ((A) % (ALIGN))))

typedef struct {
    XC_s32 reg_index;
    XC_s32 cls_index;
    XC_u32 height;
    XC_u32 width;
    XC_u32 stride;
} detect_level_info;

struct XC_image_infer_handle {
    XC_image_infer_config config;
    XC_cl_context context;
    XC_cl_device_id *devices;
    XC_cl_u32 num_devices;
    XC_cl_graph graph;
    XC_cl_tensor_info_inout input;
    XC_cl_tensor_info_inout output;
    XC_cl_tensor *input_tensor_arr;
    XC_cl_tensor *output_tensor_arr;
    XC_u64 work_phy_addr;
    XC_u64 weight_phy_addr;
    XC_u64 input_phy_addr;
    XC_u64 output_phy_addr;
    void *work_vir_addr;
    void *weight_vir_addr;
    void *input_vir_addr;
    void *output_vir_addr;
    XC_cl_u32 worksize;
    XC_cl_u32 weightsize;
    XC_u32 inputsize;
    XC_u32 outputsize;
    XC_bool sys_inited;
    XC_bool cl_inited;
};

static XC_u32 get_pixel_bytes(XC_image_format pixel_format)
{
    switch (pixel_format) {
        case XC_IMAGE_FORMAT_RGB888:
        case XC_IMAGE_FORMAT_BGR888:
            return 3;
        default:
            return 0;
    }
}

static XC_s32 mmz_alloc(XC_u64 *phy_addr, void **vir_addr, XC_u32 size)
{
    *phy_addr = XC_mmz_alloc(NULL, "xmimg", size);
    if (*phy_addr == 0) {
        return -1;
    }

    *vir_addr = XC_mmz_map(*phy_addr, size, 0);
    if (*vir_addr == NULL) {
        XC_mmz_free(*phy_addr);
        *phy_addr = 0;
        return -1;
    }

    return 0;
}

static XC_void mmz_release(XC_u64 phy_addr, void *vir_addr)
{
    if (vir_addr != NULL) {
        XC_mmz_unmap(vir_addr);
    }
    if (phy_addr != 0) {
        XC_mmz_free(phy_addr);
    }
}

static XC_s32 input_img_to_chw(const XC_input_img *input_img, XC_u8 *tensor_buffer,
    XC_u32 dst_width, XC_u32 dst_height)
{
    XC_u32 bytes_per_pixel;
    XC_u32 map_size;
    XC_u32 hw;
    XC_u8 *frame_vir_addr;
    XC_u32 x;
    XC_u32 y;

    if (input_img == NULL || tensor_buffer == NULL) {
        return -1;
    }

    bytes_per_pixel = get_pixel_bytes(input_img->pixel_format);
    if (bytes_per_pixel == 0 || input_img->phy_addr == 0 || input_img->width == 0 ||
        input_img->height == 0 || input_img->stride == 0) {
        return -1;
    }

    if (input_img->width != dst_width || input_img->height != dst_height) {
        return -1;
    }

    if (input_img->stride < input_img->width * bytes_per_pixel) {
        return -1;
    }

    map_size = input_img->stride * input_img->height;
    frame_vir_addr = (XC_u8 *)XC_mmz_map(input_img->phy_addr, map_size, 0);
    if (frame_vir_addr == NULL) {
        return -1;
    }

    hw = dst_width * dst_height;

    for (y = 0; y < dst_height; y++) {
        const XC_u8 *src_row = frame_vir_addr + y * input_img->stride;
        for (x = 0; x < dst_width; x++) {
            const XC_u8 *pixel = src_row + x * bytes_per_pixel;
            XC_u32 dst_index = y * dst_width + x;

            if (input_img->pixel_format == XC_IMAGE_FORMAT_RGB888) {
                tensor_buffer[dst_index] = pixel[0];
                tensor_buffer[hw + dst_index] = pixel[1];
                tensor_buffer[2 * hw + dst_index] = pixel[2];
            } else {
                tensor_buffer[dst_index] = pixel[2];
                tensor_buffer[hw + dst_index] = pixel[1];
                tensor_buffer[2 * hw + dst_index] = pixel[0];
            }
        }
    }

    XC_mmz_unmap(frame_vir_addr);
    return 0;
}

static XC_u32 get_tensor_bytes_per_element(const XC_cl_tensor *tensor)
{
    if (tensor->shape.type == XMEDIA_CL_FP32) {
        return sizeof(XC_float);
    }
    if (tensor->shape.type == XMEDIA_CL_UINT8) {
        return sizeof(XC_u8);
    }
    if (tensor->shape.type == XMEDIA_CL_INT8) {
        return sizeof(XC_s8);
    }

    return 0;
}

static XC_u32 get_tensor_physical_width(const XC_cl_tensor *tensor)
{
    XC_u32 channels = tensor->shape.dims[1];
    XC_u32 height = tensor->shape.dims[2];
    XC_u32 logical_width = tensor->shape.dims[3];
    XC_u32 bytes_per_elem = get_tensor_bytes_per_element(tensor);

    if (channels == 0 || height == 0 || bytes_per_elem == 0 || tensor->size == 0) {
        return logical_width;
    }

    return tensor->size / (channels * height * bytes_per_elem);
}

static XC_float tensor_value_to_float(const XC_cl_tensor *tensor, XC_u32 channel,
    XC_u32 y, XC_u32 x)
{
    XC_u32 height = tensor->shape.dims[2];
    XC_u32 width = get_tensor_physical_width(tensor);
    XC_u32 index = (channel * height + y) * width + x;

    if (tensor->shape.type == XMEDIA_CL_FP32) {
        const XC_float *data = (const XC_float *)tensor->addr;
        return data[index];
    }

    if (tensor->shape.type == XMEDIA_CL_UINT8) {
        const XC_u8 *data = (const XC_u8 *)tensor->addr;
        return ((XC_float)data[index] - tensor->quant.zp) * tensor->quant.scale;
    }

    if (tensor->shape.type == XMEDIA_CL_INT8) {
        const XC_s8 *data = (const XC_s8 *)tensor->addr;
        return ((XC_float)data[index] - tensor->quant.zp) * tensor->quant.scale;
    }

    return 0.0f;
}

static XC_float sigmoidf_safe(XC_float x)
{
    if (x >= 0.0f) {
        XC_float z = expf(-x);
        return 1.0f / (1.0f + z);
    }

    XC_float z = expf(x);
    return z / (1.0f + z);
}

static XC_float clampf_safe(XC_float value, XC_float min_value, XC_float max_value)
{
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

static XC_float decode_dfl_distance(const XC_cl_tensor *reg_tensor,
    XC_u32 side_index, XC_u32 y, XC_u32 x, XC_u32 stride)
{
    XC_float logits[XC_DFL_BINS];
    XC_float max_logit;
    XC_float sum = 0.0f;
    XC_float expectation = 0.0f;
    XC_u32 k;
    XC_u32 base_channel = side_index * XC_DFL_BINS;

    max_logit = tensor_value_to_float(reg_tensor, base_channel, y, x);
    for (k = 0; k < XC_DFL_BINS; k++) {
        logits[k] = tensor_value_to_float(reg_tensor, base_channel + k, y, x);
        if (logits[k] > max_logit) {
            max_logit = logits[k];
        }
    }

    for (k = 0; k < XC_DFL_BINS; k++) {
        logits[k] = expf(logits[k] - max_logit);
        sum += logits[k];
    }

    if (sum <= 0.0f) {
        return 0.0f;
    }

    for (k = 0; k < XC_DFL_BINS; k++) {
        expectation += ((XC_float)k) * (logits[k] / sum);
    }

    return expectation * stride;
}

static XC_s32 compare_candidate_desc(const void *left, const void *right)
{
    const XC_detect_box *a = (const XC_detect_box *)left;
    const XC_detect_box *b = (const XC_detect_box *)right;

    if (a->score < b->score) {
        return 1;
    }
    if (a->score > b->score) {
        return -1;
    }
    return 0;
}

static XC_float compute_iou(const XC_detect_box *a, const XC_detect_box *b)
{
    XC_float xx1 = a->x1 > b->x1 ? a->x1 : b->x1;
    XC_float yy1 = a->y1 > b->y1 ? a->y1 : b->y1;
    XC_float xx2 = a->x2 < b->x2 ? a->x2 : b->x2;
    XC_float yy2 = a->y2 < b->y2 ? a->y2 : b->y2;
    XC_float inter_w = xx2 - xx1;
    XC_float inter_h = yy2 - yy1;
    XC_float inter_area;
    XC_float area_a;
    XC_float area_b;
    XC_float union_area;

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

static XC_u32 apply_nms(XC_detect_box *candidates, XC_u32 candidate_count,
    XC_float iou_thresh)
{
    XC_u8 *suppressed;
    XC_u32 keep_count = 0;
    XC_u32 i;
    XC_u32 j;

    suppressed = (XC_u8 *)calloc(candidate_count, sizeof(XC_u8));
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

static XC_s32 collect_detect_levels(const XC_image_infer_handle *handle,
    detect_level_info *levels, XC_u32 max_levels)
{
    XC_u32 i;
    XC_u32 level_count = 0;
    XC_cl_tensor_info_inout *output = (XC_cl_tensor_info_inout *)&handle->output;

    for (i = 0; i < output->num; i++) {
        XC_cl_tensor *tensor = &output->tensor[i];
        XC_u32 channels = tensor->shape.dims[1];
        XC_u32 height = tensor->shape.dims[2];
        XC_u32 width = tensor->shape.dims[3];
        XC_u32 level_index;
        XC_s32 matched = -1;

        for (level_index = 0; level_index < level_count; level_index++) {
            if (levels[level_index].height == height && levels[level_index].width == width) {
                matched = (XC_s32)level_index;
                break;
            }
        }

        if (matched < 0) {
            if (level_count >= max_levels) {
                continue;
            }
            matched = (XC_s32)level_count;
            levels[level_count].reg_index = -1;
            levels[level_count].cls_index = -1;
            levels[level_count].height = height;
            levels[level_count].width = width;
            levels[level_count].stride = width == 0 ? 0 : (handle->config.image_width / width);
            level_count++;
        }

        if (channels == XC_DFL_BINS * 4) {
            levels[matched].reg_index = (XC_s32)i;
        } else if (channels == XC_YOLO_CLASS_NUM) {
            levels[matched].cls_index = (XC_s32)i;
        }
    }

    return (XC_s32)level_count;
}

static XC_s32 collect_candidate_boxes(const XC_image_infer_handle *handle,
    XC_detect_result *result)
{
    detect_level_info levels[3];
    XC_detect_box candidates[XC_MAX_CANDIDATES];
    XC_u32 candidate_count = 0;
    XC_s32 level_count;
    XC_s32 level_index;

    memset(levels, 0, sizeof(levels));
    level_count = collect_detect_levels(handle, levels, 3);
    if (level_count <= 0) {
        result->boxes = NULL;
        result->count = 0;
        return 0;
    }

    for (level_index = 0; level_index < level_count; level_index++) {
        detect_level_info *level = &levels[level_index];
        XC_cl_tensor *reg_tensor;
        XC_cl_tensor *cls_tensor;
        XC_u32 y;
        XC_u32 x;

        if (level->reg_index < 0 || level->cls_index < 0) {
            continue;
        }

        reg_tensor = &handle->output.tensor[level->reg_index];
        cls_tensor = &handle->output.tensor[level->cls_index];

        for (y = 0; y < level->height; y++) {
            for (x = 0; x < level->width; x++) {
                XC_float best_logit = tensor_value_to_float(cls_tensor, 0, y, x);
                XC_s32 best_class = 0;
                XC_float score;
                XC_float left;
                XC_float top;
                XC_float right;
                XC_float bottom;
                XC_float center_x;
                XC_float center_y;
                XC_float scale_x;
                XC_float scale_y;
                XC_float x1;
                XC_float y1;
                XC_float x2;
                XC_float y2;
                XC_u32 c;

                for (c = 1; c < XC_YOLO_CLASS_NUM; c++) {
                    XC_float logit = tensor_value_to_float(cls_tensor, c, y, x);
                    if (logit > best_logit) {
                        best_logit = logit;
                        best_class = (XC_s32)c;
                    }
                }

                score = sigmoidf_safe(best_logit);
                if (best_class != (XC_s32)handle->config.person_class_id ||
                    score < handle->config.score_thresh) {
                    continue;
                }

                left = decode_dfl_distance(reg_tensor, 0, y, x, level->stride);
                top = decode_dfl_distance(reg_tensor, 1, y, x, level->stride);
                right = decode_dfl_distance(reg_tensor, 2, y, x, level->stride);
                bottom = decode_dfl_distance(reg_tensor, 3, y, x, level->stride);

                center_x = ((XC_float)x + 0.5f) * level->stride;
                center_y = ((XC_float)y + 0.5f) * level->stride;
                scale_x = level->width == 0 ? 1.0f :
                    ((XC_float)handle->config.image_width / (level->width * level->stride));
                scale_y = 1.0f;

                x1 = clampf_safe((center_x - left) * scale_x, 0.0f,
                    (XC_float)(handle->config.image_width - 1));
                y1 = clampf_safe((center_y - top) * scale_y, 0.0f,
                    (XC_float)(handle->config.image_height - 1));
                x2 = clampf_safe((center_x + right) * scale_x, 0.0f,
                    (XC_float)(handle->config.image_width - 1));
                y2 = clampf_safe((center_y + bottom) * scale_y, 0.0f,
                    (XC_float)(handle->config.image_height - 1));

                if ((x2 <= x1) || (y2 <= y1) || candidate_count >= XC_MAX_CANDIDATES) {
                    continue;
                }

                candidates[candidate_count].class_id = best_class;
                candidates[candidate_count].score = score;
                candidates[candidate_count].x1 = x1;
                candidates[candidate_count].y1 = y1;
                candidates[candidate_count].x2 = x2;
                candidates[candidate_count].y2 = y2;
                candidates[candidate_count].stride = level->stride;
                candidates[candidate_count].level = (XC_u32)level_index;
                candidate_count++;
            }
        }
    }

    if (candidate_count == 0) {
        result->boxes = NULL;
        result->count = 0;
        return 0;
    }

    qsort(candidates, candidate_count, sizeof(XC_detect_box), compare_candidate_desc);
    candidate_count = apply_nms(candidates, candidate_count, handle->config.nms_iou_thresh);

    result->boxes = (XC_detect_box *)calloc(candidate_count, sizeof(XC_detect_box));
    if (result->boxes == NULL) {
        result->count = 0;
        return -1;
    }

    memcpy(result->boxes, candidates, candidate_count * sizeof(XC_detect_box));
    result->count = candidate_count;
    return 0;
}

static XC_s32 assign_tensor_addrs(XC_cl_tensor_info_inout *tensor_info, void *base_addr)
{
    XC_s32 i;

    for (i = 0; i < (XC_s32)tensor_info->num; i++) {
        if (i > 0) {
            tensor_info->tensor[i].addr = (XC_u8 *)tensor_info->tensor[i - 1].addr +
                XC_ALIGN_FUNC(tensor_info->tensor[i - 1].size, XC_ALIGN_BYTE);
        } else {
            tensor_info->tensor[i].addr = base_addr;
        }
    }

    return 0;
}

static XC_void fill_default_config(XC_image_infer_config *config)
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

XC_s32 XC_image_infer_init(const XC_image_infer_config *config,
    XC_image_infer_handle **handle)
{
    XC_image_infer_handle *ctx;
    XC_cl_s32 err_code = 0;
    XC_s32 ret;
    XC_s32 i;

    if (config == NULL || handle == NULL || config->model_path == NULL) {
        return -1;
    }

    *handle = NULL;
    ctx = (XC_image_infer_handle *)calloc(1, sizeof(*ctx));
    if (ctx == NULL) {
        return -1;
    }

    ctx->config = *config;
    fill_default_config(&ctx->config);

    ret = XC_sys_init(NULL);
    if (ret != 0) {
        XC_image_infer_destroy(ctx);
        return ret;
    }
    ctx->sys_inited = XMEDIA_TRUE;

    ret = XC_cl_init();
    if (ret != 0) {
        XC_image_infer_destroy(ctx);
        return ret;
    }
    ctx->cl_inited = XMEDIA_TRUE;

    ret = XC_cl_get_device_ids(XMEDIA_CL_DEVICE_ALL, NULL, &ctx->num_devices);
    if (ret != 0 || ctx->num_devices == 0) {
        XC_image_infer_destroy(ctx);
        return ret != 0 ? ret : -1;
    }

    ctx->devices = (XC_cl_device_id *)calloc(ctx->num_devices, sizeof(XC_cl_device_id));
    if (ctx->devices == NULL) {
        XC_image_infer_destroy(ctx);
        return -1;
    }

    ret = XC_cl_get_device_ids(XMEDIA_CL_DEVICE_ALL, ctx->devices, &ctx->num_devices);
    if (ret != 0) {
        XC_image_infer_destroy(ctx);
        return ret;
    }

    ctx->context = XC_cl_create_context(ctx->num_devices, ctx->devices, &err_code);
    if (err_code != 0 || ctx->context == NULL) {
        XC_image_infer_destroy(ctx);
        return err_code != 0 ? err_code : -1;
    }

    ret = XC_cl_graph_querysize_from_file(ctx->config.model_path, &ctx->worksize, &ctx->weightsize);
    if (ret != 0) {
        XC_image_infer_destroy(ctx);
        return ret;
    }

    if (ctx->worksize > 0 && mmz_alloc(&ctx->work_phy_addr, &ctx->work_vir_addr, ctx->worksize) != 0) {
        XC_image_infer_destroy(ctx);
        return -1;
    }

    if (ctx->weightsize > 0 && mmz_alloc(&ctx->weight_phy_addr, &ctx->weight_vir_addr, ctx->weightsize) != 0) {
        XC_image_infer_destroy(ctx);
        return -1;
    }

    ret = XC_cl_graph_loadmodel_from_file_withmem(&ctx->context,
        ctx->config.model_path, ctx->work_vir_addr, ctx->worksize,
        ctx->weight_vir_addr, ctx->weightsize, &ctx->graph);
    if (ret != 0) {
        XC_image_infer_destroy(ctx);
        return ret;
    }

    ret = XC_cl_graph_get_input(ctx->graph, 0, &ctx->input);
    if (ret != 0) {
        XC_image_infer_destroy(ctx);
        return ret;
    }

    ctx->input_tensor_arr = (XC_cl_tensor *)calloc(ctx->input.num, sizeof(XC_cl_tensor));
    if (ctx->input_tensor_arr == NULL) {
        XC_image_infer_destroy(ctx);
        return -1;
    }
    ctx->input.tensor = ctx->input_tensor_arr;

    ret = XC_cl_graph_get_input(ctx->graph, ctx->input.num, &ctx->input);
    if (ret != 0) {
        XC_image_infer_destroy(ctx);
        return ret;
    }

    ret = XC_cl_graph_get_output(ctx->graph, 0, &ctx->output);
    if (ret != 0) {
        XC_image_infer_destroy(ctx);
        return ret;
    }

    ctx->output_tensor_arr = (XC_cl_tensor *)calloc(ctx->output.num, sizeof(XC_cl_tensor));
    if (ctx->output_tensor_arr == NULL) {
        XC_image_infer_destroy(ctx);
        return -1;
    }
    ctx->output.tensor = ctx->output_tensor_arr;

    ret = XC_cl_graph_get_output(ctx->graph, ctx->output.num, &ctx->output);
    if (ret != 0) {
        XC_image_infer_destroy(ctx);
        return ret;
    }

    for (i = 0; i < (XC_s32)ctx->input.num; i++) {
        ctx->inputsize += XC_ALIGN_FUNC(ctx->input.tensor[i].size, XC_ALIGN_BYTE);
    }
    for (i = 0; i < (XC_s32)ctx->output.num; i++) {
        ctx->outputsize += XC_ALIGN_FUNC(ctx->output.tensor[i].size, XC_ALIGN_BYTE);
    }

    if (ctx->inputsize > 0 && mmz_alloc(&ctx->input_phy_addr, &ctx->input_vir_addr, ctx->inputsize) != 0) {
        XC_image_infer_destroy(ctx);
        return -1;
    }

    if (ctx->outputsize > 0 && mmz_alloc(&ctx->output_phy_addr, &ctx->output_vir_addr, ctx->outputsize) != 0) {
        XC_image_infer_destroy(ctx);
        return -1;
    }

    assign_tensor_addrs(&ctx->input, ctx->input_vir_addr);
    assign_tensor_addrs(&ctx->output, ctx->output_vir_addr);

    *handle = ctx;
    return 0;
}

XC_s32 XC_image_infer_detect(XC_image_infer_handle *handle,
    const XC_input_img *input_img, XC_detect_result *result)
{
    XC_s32 ret;
    XC_u32 channel_size;

    if (handle == NULL || input_img == NULL || result == NULL) {
        return -1;
    }

    result->boxes = NULL;
    result->count = 0;

    if (handle->input.num == 0 || handle->input.tensor[0].addr == NULL) {
        return -1;
    }

    channel_size = handle->config.image_width * handle->config.image_height;
    if (channel_size * 3 > handle->input.tensor[0].size) {
        return -1;
    }

    memset(handle->input.tensor[0].addr, 0, handle->input.tensor[0].size);
    ret = input_img_to_chw(input_img, (XC_u8 *)handle->input.tensor[0].addr,
        handle->config.image_width, handle->config.image_height);
    if (ret != 0) {
        return ret;
    }

    ret = XC_cl_graph_set_inout(handle->graph, &handle->input, &handle->output);
    if (ret != 0) {
        return ret;
    }

    ret = XC_cl_graph_process(handle->graph);
    if (ret != 0) {
        return ret;
    }

    return collect_candidate_boxes(handle, result);
}

XC_void XC_image_infer_result_deinit(XC_detect_result *result)
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

XC_void XC_image_infer_destroy(XC_image_infer_handle *handle)
{
    XC_s32 err;

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
        err = XC_cl_graph_unload(handle->graph);
        if (err != 0) {
            printf("XC_cl_graph_unload err, errno %d\n", err);
        }
    }

    mmz_release(handle->weight_phy_addr, handle->weight_vir_addr);
    mmz_release(handle->work_phy_addr, handle->work_vir_addr);

    if (handle->context != NULL) {
        err = XC_cl_release_context(handle->context);
        if (err != 0) {
            printf("XC_cl_release_context err, errno %d\n", err);
        }
    }

    if (handle->devices != NULL) {
        err = XC_cl_release_device_ids(handle->devices, &handle->num_devices);
        if (err != 0) {
            printf("XC_cl_release_device_ids err, errno %d\n", err);
        }
        free(handle->devices);
    }

    if (handle->cl_inited == XMEDIA_TRUE) {
        err = XC_cl_uninit();
        if (err != 0) {
            printf("XC_cl_uninit err, errno %d\n", err);
        }
    }

    if (handle->sys_inited == XMEDIA_TRUE) {
        err = XC_sys_exit();
        if (err != 0) {
            printf("XC_sys_exit err, errno %d\n", err);
        }
    }

    free(handle);
}
