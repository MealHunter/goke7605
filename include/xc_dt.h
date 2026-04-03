#ifndef __XC_DT_H__
#define __XC_DT_H__

#include "xmedia_cl.h"
#include "xc_common_datatype.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 前向声明：对外只暴露句柄指针，内部实现细节隐藏在 .cpp 中。
 * 上层只需要持有并传递该句柄，不应直接访问其成员。
 */

typedef struct XC_image_infer_handle XC_image_infer_handle;

/*
 * 推理初始化参数。
 *
 * model_path:
 *   .xmm 模型文件路径。
 * image_width / image_height:
 *   模型期望输入尺寸。当前实现要求外部传入帧尺寸与这里保持一致，
 *   不在库内部做 resize / letterbox。
 * score_thresh:
 *   置信度阈值，低于该阈值的候选框会被过滤。
 * nms_iou_thresh:
 *   NMS 阈值，用于去除重叠过高的框。
 * person_class_id:
 *   当前实现只保留该类别的输出，通常用于 person 类单类检测场景。
 */
typedef struct {
    const char *model_path;
    XC_U32 image_width;
    XC_U32 image_height;
    XC_FLOAT score_thresh;
    XC_FLOAT nms_iou_thresh;
    XC_U32 person_class_id;
} XC_image_infer_config;

/*
 * 单个检测框结果。
 *
 * class_id / score:
 *   类别编号与置信度。
 * x1/y1/x2/y2:
 *   检测框左上角/右下角坐标，坐标系基于输入图像尺寸。
 */
typedef struct {
    XC_S32 class_id;
    XC_FLOAT score;
    XC_FLOAT x1;
    XC_FLOAT y1;
    XC_FLOAT x2;
    XC_FLOAT y2;
} XC_detect_box;

/*
 * 输入图像像素格式。
 * 当前仅支持 3 通道打包格式，由 detect 阶段转换为 NPU 需要的 CHW 布局。
 */
typedef enum {
    XC_IMAGE_FORMAT_RGB888 = 0,
    XC_IMAGE_FORMAT_BGR888 = 1,
} XC_image_format;

/*
 * 单帧输入描述。
 *
 * phy_addr:
 *   图像帧所在物理地址。库内部会通过 MMZ map 读取数据。
 * width / height:
 *   输入帧宽高，必须与初始化配置中的模型输入尺寸一致。
 * stride:
 *   每行字节跨度，允许大于 width * 3，用于兼容硬件对齐后的图像缓存。
 * pixel_format:
 *   当前帧的颜色通道顺序。
 */
typedef struct {
    XC_U64 phy_addr;
    XC_U32 width;
    XC_U32 height;
    XC_U32 stride;
    XC_image_format pixel_format;
} XC_input_img;

/*
 * 检测结果集合。
 *
 * boxes 内存由库内部申请，调用方在使用完成后必须调用
 * XC_image_infer_result_deinit() 释放。
 */
typedef struct {
    XC_detect_box *boxes;
    XC_U32 count;
} XC_detect_result;

/*
 * 初始化推理句柄并加载模型。
 *
 * 成功后会完成：系统初始化、CL 运行时初始化、模型加载、
 * 输入输出 tensor 信息查询，以及推理所需 MMZ 内存分配。
 */
XC_S32 XC_image_infer_init(const XC_image_infer_config *config, XC_image_infer_handle **handle);

/*
 * 对单帧图像执行检测。
 *
 * 输入为物理地址描述的 RGB/BGR 图像帧，输出为过滤并经过 NMS 的检测框集合。
 */
XC_S32 XC_image_infer_detect(XC_image_infer_handle *handle, const XC_input_img *input_img, XC_detect_result *result);

/*
 * 释放一次 detect 调用产生的结果缓存。
 */
void XC_image_infer_result_deinit(XC_detect_result *result);

/*
 * 销毁推理句柄并释放所有底层资源。
 *
 * 包括 graph、输入输出 tensor 缓冲区、work/weight 内存、
 * context、device 列表，以及系统/CL 运行时状态。
 */
void XC_image_infer_destroy(XC_image_infer_handle *handle);

#ifdef __cplusplus
}
#endif

#endif
