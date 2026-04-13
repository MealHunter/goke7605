#include "xc_dt.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "xmedia_cl.h"
#include "xc_common_datatype.h"
#include "xmedia_mmz.h"
#include "xmedia_sys.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 这个文件实现一个“板端单帧目标检测封装层”。
 * 整体流程如下：
 * 1. init 阶段初始化系统与 xmedia_cl，加载 .xmm 模型，准备输入输出 tensor；
 * 2. detect 阶段把物理地址图像搬运为 CHW 输入，执行 graph 推理；
 * 3. 从输出 tensor 中解码候选框，按类别过滤并做 NMS；
 * 4. destroy 阶段按逆序释放所有底层资源。
 */

#define XC_ALIGN_BYTE 16
#define XC_DFL_BINS 16
#define XC_YOLO_CLASS_NUM 80
#define XC_MAX_CANDIDATES 6000

#define XC_ALIGN_FUNC(A, ALIGN) ((((A) % (ALIGN)) == 0) ? (A) : ((A) + (ALIGN) - ((A) % (ALIGN))))

typedef struct {
    XC_S32 reg_index;
    XC_S32 cls_index;
    XC_U32 height;
    XC_U32 width;
    XC_U32 stride;
} detect_level_info;

/*
 * 推理句柄内部状态。
 *
 * 对外只暴露一个不透明指针，对内保存：
 * - 初始化配置
 * - xmedia_cl 设备 / context / graph
 * - 输入输出 tensor 元信息与地址数组
 * - work / weight / input / output 四类 MMZ 物理+虚拟地址
 * - 运行时初始化标记，便于 destroy 时安全回收
 */
struct XC_image_infer_handle {
    XC_image_infer_config config;
    xmedia_cl_context context;
    xmedia_cl_device_id *devices;
    xmedia_cl_u32 num_devices;
    xmedia_cl_graph graph;              //只是一个地址
    xmedia_cl_tensor_info_inout input;
    xmedia_cl_tensor_info_inout output;
    xmedia_cl_tensor *input_tensor_arr;
    xmedia_cl_tensor *output_tensor_arr;
    XC_U64 work_phy_addr;
    XC_U64 weight_phy_addr;
    XC_U64 input_phy_addr;
    XC_U64 output_phy_addr;
    void *work_vir_addr;
    void *weight_vir_addr;
    void *input_vir_addr;
    void *output_vir_addr;
    xmedia_cl_u32 worksize;
    xmedia_cl_u32 weightsize;
    XC_U32 inputsize;
    XC_U32 outputsize;
    XC_BOOL sys_inited;
    XC_BOOL cl_inited;
};

static XC_U32 get_pixel_bytes(XC_image_format pixel_format)
{
    switch (pixel_format) {
        case XC_IMAGE_FORMAT_RGB888:
        case XC_IMAGE_FORMAT_BGR888:
            return 3;
        default:
            return 0;
    }
}

/*
 * 申请一段 MMZ 物理内存并立即映射到虚拟地址。
 * 这是整个推理流程里最常用的底层内存分配封装。
 */
static XC_S32 mmz_alloc(XC_U64 *phy_addr, void **vir_addr, XC_U32 size)
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

/*
 * 释放 MMZ 资源。
 * 允许 phy_addr 或 vir_addr 单独为空，方便 destroy 做幂等回收。
 */
static XC_void mmz_release(XC_U64 phy_addr, void *vir_addr)
{
    if (vir_addr != NULL) {
        XC_mmz_unmap(vir_addr);
    }
    if (phy_addr != 0) {
        XC_mmz_free(phy_addr);
    }
}

/*
 * 将一帧板端图像从物理地址映射出来，并转换为模型输入所需的 CHW 布局。
 *
 * 当前限制：
 * - 只支持 RGB888 / BGR888；
 * - 输入尺寸必须与模型输入尺寸完全一致；
 * - 不负责缩放、裁剪和颜色空间转换之外的额外预处理。
 */
static XC_S32 input_img_to_chw(const XC_input_img *input_img, XC_U8 *tensor_buffer,
    XC_U32 dst_width, XC_U32 dst_height)
{
    XC_U32 bytes_per_pixel;
    XC_U32 map_size;
    XC_U32 hw;
    XC_U8 *frame_vir_addr;
    XC_U32 x;
    XC_U32 y;

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
    frame_vir_addr = (XC_U8 *)XC_mmz_map(input_img->phy_addr, map_size, 0);
    if (frame_vir_addr == NULL) {
        return -1;
    }

    hw = dst_width * dst_height;

    for (y = 0; y < dst_height; y++) {
        const XC_U8 *src_row = frame_vir_addr + y * input_img->stride;
        for (x = 0; x < dst_width; x++) {
            const XC_U8 *pixel = src_row + x * bytes_per_pixel;
            XC_U32 dst_index = y * dst_width + x;

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

/*
 * 返回 tensor 每个元素占用字节数。
 * 用于后续根据 tensor->size 反推出实际物理宽度，避免按逻辑 shape 直接索引。
 */
static XC_U32 get_tensor_bytes_per_element(const xmedia_cl_tensor *tensor)
{
    if (tensor->shape.type == XMEDIA_CL_FP32) {
        return sizeof(XC_FLOAT);
    }
    if (tensor->shape.type == XMEDIA_CL_UINT8) {
        return sizeof(XC_U8);
    }
    if (tensor->shape.type == XMEDIA_CL_INT8) {
        return sizeof(XC_S8);
    }

    return 0;
}

/*
 * 根据 tensor 总字节数推导输出 tensor 的“物理宽度”。
 *
 * 某些 NPU 输出张量在内存里会按硬件要求进行行对齐，
 * 实际宽度可能大于 shape.dims[3] 的逻辑宽度。
 * 后续读取 tensor 时必须按物理宽度访问，否则从第二行开始可能取错数据。
 */
static XC_U32 get_tensor_physical_width(const xmedia_cl_tensor *tensor)
{
    XC_U32 channels = tensor->shape.dims[1];
    XC_U32 height = tensor->shape.dims[2];
    XC_U32 logical_width = tensor->shape.dims[3];
    XC_U32 bytes_per_elem = get_tensor_bytes_per_element(tensor);

    if (channels == 0 || height == 0 || bytes_per_elem == 0 || tensor->size == 0) {
        return logical_width;
    }

    return tensor->size / (channels * height * bytes_per_elem);
}

/*
 * 读取 tensor 中指定位置的值，并统一转成 float。
 * 同时处理 FP32 / UINT8 / INT8 三种张量类型及量化反算。
 */
static XC_FLOAT tensor_value_to_float(const xmedia_cl_tensor *tensor, XC_U32 channel,
    XC_U32 y, XC_U32 x)
{
    XC_U32 height = tensor->shape.dims[2];
    XC_U32 width = get_tensor_physical_width(tensor);
    XC_U32 index = (channel * height + y) * width + x;

    if (tensor->shape.type == XMEDIA_CL_FP32) {
        const XC_FLOAT *data = (const XC_FLOAT *)tensor->addr;
        return data[index];
    }

    if (tensor->shape.type == XMEDIA_CL_UINT8) {
        const XC_U8 *data = (const XC_U8 *)tensor->addr;
        return ((XC_FLOAT)data[index] - tensor->quant.zp) * tensor->quant.scale;
    }

    if (tensor->shape.type == XMEDIA_CL_INT8) {
        const XC_S8 *data = (const XC_S8 *)tensor->addr;
        return ((XC_FLOAT)data[index] - tensor->quant.zp) * tensor->quant.scale;
    }

    return 0.0f;
}

/* 稳定版 sigmoid，避免大正/大负输入时数值溢出。 */
static XC_FLOAT sigmoidf_safe(XC_FLOAT x)
{
    if (x >= 0.0f) {
        XC_FLOAT z = expf(-x);
        return 1.0f / (1.0f + z);
    }

    XC_FLOAT z = expf(x);
    return z / (1.0f + z);
}

/* 将浮点数限制在给定区间，主要用于输出框坐标裁剪。 */
static XC_FLOAT clampf_safe(XC_FLOAT value, XC_FLOAT min_value, XC_FLOAT max_value)
{
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

/*
 * DFL(Distribution Focal Loss) 距离解码。
 *
 * 回归头对每个边界不是直接输出一个距离，而是输出若干 bin 的分布。
 * 这里先做 softmax，再计算期望值，最后乘以 stride 还原到当前检测层尺度。
 */
static XC_FLOAT decode_dfl_distance(const xmedia_cl_tensor *reg_tensor,
    XC_U32 side_index, XC_U32 y, XC_U32 x, XC_U32 stride)
{
    XC_FLOAT logits[XC_DFL_BINS];
    XC_FLOAT max_logit;
    XC_FLOAT sum = 0.0f;
    XC_FLOAT expectation = 0.0f;
    XC_U32 k;
    XC_U32 base_channel = side_index * XC_DFL_BINS;

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
        expectation += ((XC_FLOAT)k) * (logits[k] / sum);
    }

    return expectation * stride;
}

/* qsort 比较函数：按 score 从大到小排序。 */
static XC_S32 compare_candidate_desc(const void *left, const void *right)
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

/* 计算两个检测框的 IoU，用于后续 NMS。 */
static XC_FLOAT compute_iou(const XC_detect_box *a, const XC_detect_box *b)
{
    XC_FLOAT xx1 = a->x1 > b->x1 ? a->x1 : b->x1;
    XC_FLOAT yy1 = a->y1 > b->y1 ? a->y1 : b->y1;
    XC_FLOAT xx2 = a->x2 < b->x2 ? a->x2 : b->x2;
    XC_FLOAT yy2 = a->y2 < b->y2 ? a->y2 : b->y2;
    XC_FLOAT inter_w = xx2 - xx1;
    XC_FLOAT inter_h = yy2 - yy1;
    XC_FLOAT inter_area;
    XC_FLOAT area_a;
    XC_FLOAT area_b;
    XC_FLOAT union_area;

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
 * 对候选框做按类别 NMS。
 * 输入数组会被原地压缩，返回保留下来的框数量。
 */
static XC_U32 apply_nms(XC_detect_box *candidates, XC_U32 candidate_count,
    XC_FLOAT iou_thresh)
{
    XC_U8 *suppressed;
    XC_U32 keep_count = 0;
    XC_U32 i;
    XC_U32 j;

    suppressed = (XC_U8 *)calloc(candidate_count, sizeof(XC_U8));
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

/*
 * 根据输出 tensor 的 shape 将多个输出分组成检测层。
 *
 * 当前默认每个检测层由一组回归头 + 一组分类头组成，
 * 并通过通道数判断：
 * - XC_DFL_BINS * 4 -> 回归头
 * - XC_YOLO_CLASS_NUM -> 分类头
 */
static XC_S32 collect_detect_levels(const XC_image_infer_handle *handle,
    detect_level_info *levels, XC_U32 max_levels)
{
    XC_U32 i;
    XC_U32 level_count = 0;
    xmedia_cl_tensor_info_inout *output = (xmedia_cl_tensor_info_inout *)&handle->output;

    for (i = 0; i < output->num; i++) {
        xmedia_cl_tensor *tensor = &output->tensor[i];
        XC_U32 channels = tensor->shape.dims[1];
        XC_U32 height = tensor->shape.dims[2];
        XC_U32 width = tensor->shape.dims[3];
        XC_U32 level_index;
        XC_S32 matched = -1;

        for (level_index = 0; level_index < level_count; level_index++) {
            if (levels[level_index].height == height && levels[level_index].width == width) {
                matched = (XC_S32)level_index;
                break;
            }
        }

        if (matched < 0) {
            if (level_count >= max_levels) {
                continue;
            }
            matched = (XC_S32)level_count;
            levels[level_count].reg_index = -1;
            levels[level_count].cls_index = -1;
            levels[level_count].height = height;
            levels[level_count].width = width;
            levels[level_count].stride = width == 0 ? 0 : (handle->config.image_width / width);
            level_count++;
        }

        if (channels == XC_DFL_BINS * 4) {
            levels[matched].reg_index = (XC_S32)i;
        } else if (channels == XC_YOLO_CLASS_NUM) {
            levels[matched].cls_index = (XC_S32)i;
        }
    }

    return (XC_S32)level_count;
}

/*
 * 从模型输出 tensor 中解码出最终检测结果。
 *
 * 主要步骤：
 * 1. 识别每个检测层的 reg/cls tensor；
 * 2. 遍历所有 grid，找出得分最高类别；
 * 3. 只保留配置中的目标类别，并按阈值过滤；
 * 4. 用 DFL 结果解码 l/t/r/b；
 * 5. 转成图像坐标并裁剪；
 * 6. 按 score 排序并执行 NMS；
 * 7. 把最终结果拷贝到 result->boxes。
 */
static XC_S32 collect_candidate_boxes(const XC_image_infer_handle *handle,
    XC_detect_result *result)
{
    detect_level_info levels[3];
    XC_detect_box candidates[XC_MAX_CANDIDATES];
    XC_U32 candidate_count = 0;
    XC_S32 level_count;
    XC_S32 level_index;

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
        XC_U32 y;
        XC_U32 x;

        if (level->reg_index < 0 || level->cls_index < 0) {
            continue;
        }

        reg_tensor = &handle->output.tensor[level->reg_index];
        cls_tensor = &handle->output.tensor[level->cls_index];

        for (y = 0; y < level->height; y++) {
            for (x = 0; x < level->width; x++) {
                XC_FLOAT best_logit = tensor_value_to_float(cls_tensor, 0, y, x);
                XC_S32 best_class = 0;
                XC_FLOAT score;
                XC_FLOAT left;
                XC_FLOAT top;
                XC_FLOAT right;
                XC_FLOAT bottom;
                XC_FLOAT center_x;
                XC_FLOAT center_y;
                XC_FLOAT scale_x;
                XC_FLOAT scale_y;
                XC_FLOAT x1;
                XC_FLOAT y1;
                XC_FLOAT x2;
                XC_FLOAT y2;
                XC_U32 c;

                for (c = 1; c < XC_YOLO_CLASS_NUM; c++) {
                    XC_FLOAT logit = tensor_value_to_float(cls_tensor, c, y, x);
                    if (logit > best_logit) {
                        best_logit = logit;
                        best_class = (XC_S32)c;
                    }
                }

                score = sigmoidf_safe(best_logit);
                if (best_class != (XC_S32)handle->config.person_class_id ||
                    score < handle->config.score_thresh) {
                    continue;
                }

                left = decode_dfl_distance(reg_tensor, 0, y, x, level->stride);
                top = decode_dfl_distance(reg_tensor, 1, y, x, level->stride);
                right = decode_dfl_distance(reg_tensor, 2, y, x, level->stride);
                bottom = decode_dfl_distance(reg_tensor, 3, y, x, level->stride);

                center_x = ((XC_FLOAT)x + 0.5f) * level->stride;
                center_y = ((XC_FLOAT)y + 0.5f) * level->stride;
                scale_x = level->width == 0 ? 1.0f :
                    ((XC_FLOAT)handle->config.image_width / (level->width * level->stride));
                scale_y = level->height == 0 ? 1.0f :
                    ((XC_FLOAT)handle->config.image_height / (level->height * level->stride));

                x1 = clampf_safe((center_x - left) * scale_x, 0.0f,
                    (XC_FLOAT)(handle->config.image_width - 1));
                y1 = clampf_safe((center_y - top) * scale_y, 0.0f,
                    (XC_FLOAT)(handle->config.image_height - 1));
                x2 = clampf_safe((center_x + right) * scale_x, 0.0f,
                    (XC_FLOAT)(handle->config.image_width - 1));
                y2 = clampf_safe((center_y + bottom) * scale_y, 0.0f,
                    (XC_FLOAT)(handle->config.image_height - 1));

                if ((x2 <= x1) || (y2 <= y1) || candidate_count >= XC_MAX_CANDIDATES) {
                    continue;
                }

                candidates[candidate_count].class_id = best_class;
                candidates[candidate_count].score = score;
                candidates[candidate_count].x1 = x1;
                candidates[candidate_count].y1 = y1;
                candidates[candidate_count].x2 = x2;
                candidates[candidate_count].y2 = y2;
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

/*
 * 将一组 tensor 的 addr 指针顺序映射到连续的大块缓冲区上。
 * 相邻 tensor 之间按 16 字节对齐，便于匹配底层硬件要求。
 */
static XC_S32 assign_tensor_addrs(xmedia_cl_tensor_info_inout *tensor_info, void *base_addr)
{
    XC_S32 i;

    for (i = 0; i < (XC_S32)tensor_info->num; i++) {
        if (i > 0) {
            tensor_info->tensor[i].addr = (XC_U8 *)tensor_info->tensor[i - 1].addr +
                XC_ALIGN_FUNC(tensor_info->tensor[i - 1].size, XC_ALIGN_BYTE);
        } else {
            tensor_info->tensor[i].addr = base_addr;
        }
    }

    return 0;
}

/*
 * 填充默认配置。
 * 若调用方未显式指定常用参数，则使用当前模型的默认推理尺寸与阈值。
 */
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

/*
 * 初始化推理句柄。
 *
 * 资源准备顺序：
 * 1. 参数校验并复制配置；
 * 2. 初始化系统与 CL 运行时；
 * 3. 枚举设备并创建 context；
 * 4. 查询模型 work/weight 大小并申请 MMZ；
 * 5. 加载 graph；
 * 6. 查询输入输出 tensor 信息；
 * 7. 为输入输出 tensor 申请连续缓冲区并分配地址。
 *
 * 任一步骤失败都会调用 destroy 做统一清理。
 */
XC_S32 XC_image_infer_init(const XC_image_infer_config *config, XC_image_infer_handle **handle)
{
    XC_image_infer_handle *ctx;
    xmedia_cl_s32 err_code = 0;
    XC_S32 ret;
    XC_S32 i;

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

    ret = XC_Sys_init(NULL);
    if (ret != 0) {
        XC_image_infer_destroy(ctx);
        return ret;
    }
    ctx->sys_inited = XMEDIA_TRUE;

    ret = xmedia_cl_init();
    if (ret != 0) {
        XC_image_infer_destroy(ctx);
        return ret;
    }
    ctx->cl_inited = XMEDIA_TRUE;

    ret = xmedia_cl_get_device_ids(XMEDIA_CL_DEVICE_ALL, NULL, &ctx->num_devices);
    if (ret != 0 || ctx->num_devices == 0) {
        XC_image_infer_destroy(ctx);
        return ret != 0 ? ret : -1;
    }

    ctx->devices = (xmedia_cl_device_id *)calloc(ctx->num_devices, sizeof(xmedia_cl_device_id));
    if (ctx->devices == NULL) {
        XC_image_infer_destroy(ctx);
        return -1;
    }

    ret = xmedia_cl_get_device_ids(XMEDIA_CL_DEVICE_ALL, ctx->devices, &ctx->num_devices);
    if (ret != 0) {
        XC_image_infer_destroy(ctx);
        return ret;
    }

    ctx->context = xmedia_cl_create_context(ctx->num_devices, ctx->devices, &err_code);
    if (err_code != 0 || ctx->context == NULL) {
        XC_image_infer_destroy(ctx);
        return err_code != 0 ? err_code : -1;
    }

    ret = xmedia_cl_graph_querysize_from_file(ctx->config.model_path, &ctx->worksize, &ctx->weightsize);
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

    ret = xmedia_cl_graph_loadmodel_from_file_withmem(&ctx->context,
        ctx->config.model_path, ctx->work_vir_addr, ctx->worksize,
        ctx->weight_vir_addr, ctx->weightsize, &ctx->graph);
    if (ret != 0) {
        XC_image_infer_destroy(ctx);
        return ret;
    }

    ret = xmedia_cl_graph_get_input(ctx->graph, 0, &ctx->input);
    if (ret != 0) {
        XC_image_infer_destroy(ctx);
        return ret;
    }

    ctx->input_tensor_arr = (xmedia_cl_tensor *)calloc(ctx->input.num, sizeof(xmedia_cl_tensor));
    if (ctx->input_tensor_arr == NULL) {
        XC_image_infer_destroy(ctx);
        return -1;
    }
    ctx->input.tensor = ctx->input_tensor_arr;

    ret = xmedia_cl_graph_get_input(ctx->graph, ctx->input.num, &ctx->input);
    if (ret != 0) {
        XC_image_infer_destroy(ctx);
        return ret;
    }

    ret = xmedia_cl_graph_get_output(ctx->graph, 0, &ctx->output);
    if (ret != 0) {
        XC_image_infer_destroy(ctx);
        return ret;
    }

    ctx->output_tensor_arr = (xmedia_cl_tensor *)calloc(ctx->output.num, sizeof(xmedia_cl_tensor));
    if (ctx->output_tensor_arr == NULL) {
        XC_image_infer_destroy(ctx);
        return -1;
    }
    ctx->output.tensor = ctx->output_tensor_arr;

    ret = xmedia_cl_graph_get_output(ctx->graph, ctx->output.num, &ctx->output);
    if (ret != 0) {
        XC_image_infer_destroy(ctx);
        return ret;
    }

    for (i = 0; i < (XC_S32)ctx->input.num; i++) {
        ctx->inputsize += XC_ALIGN_FUNC(ctx->input.tensor[i].size, XC_ALIGN_BYTE);
    }
    for (i = 0; i < (XC_S32)ctx->output.num; i++) {
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

/*
 * 执行一次单帧检测。
 *
 * 主流程：
 * 1. 检查输入与 tensor 缓冲区状态；
 * 2. 将物理地址图像搬运为 CHW 输入；
 * 3. 绑定输入输出 tensor；
 * 4. 执行 graph 推理；
 * 5. 从输出 tensor 解码候选框并返回。
 */
XC_S32 XC_image_infer_detect(XC_image_infer_handle *handle, const XC_input_img *input_img, XC_detect_result *result)
{
    XC_S32 ret;
    XC_U32 channel_size;

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
    ret = input_img_to_chw(input_img, (XC_U8 *)handle->input.tensor[0].addr,
        handle->config.image_width, handle->config.image_height);
    if (ret != 0) {
        return ret;
    }

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

/* 释放一次检测调用返回的框数组。 */
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

/*
 * 销毁句柄并按依赖逆序释放资源。
 *
 * 释放顺序大致为：
 * - 输出/输入缓冲区
 * - tensor 数组
 * - graph
 * - work/weight 内存
 * - context / device
 * - cl runtime / system
 */
XC_void XC_image_infer_destroy(XC_image_infer_handle *handle)
{
    XC_S32 err;

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
        err = XC_Sys_exit();
        if (err != 0) {
            printf("XC_Sys_exit err, errno %d\n", err);
        }
    }

    free(handle);
}

#ifdef __cplusplus
}
#endif
