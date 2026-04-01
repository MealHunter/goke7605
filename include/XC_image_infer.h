#ifndef XM_IMAGE_INFER_H
#define XM_IMAGE_INFER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "xmedia_cl.h"

typedef struct XC_image_infer_handle XC_image_infer_handle;

typedef struct {
    const XC_char *model_path;
    XC_u32 image_width;
    XC_u32 image_height;
    XC_float score_thresh;
    XC_float nms_iou_thresh;
    XC_u32 person_class_id;
} XC_image_infer_config;

typedef struct {
    XC_s32 class_id;
    XC_float score;
    XC_float x1;
    XC_float y1;
    XC_float x2;
    XC_float y2;
    XC_u32 stride;
    XC_u32 level;
} XC_detect_box;

typedef enum {
    XC_IMAGE_FORMAT_RGB888 = 0,
    XC_IMAGE_FORMAT_BGR888 = 1,
} XC_image_format;

typedef struct {
    XC_u64 phy_addr;
    XC_u32 width;
    XC_u32 height;
    XC_u32 stride;
    XC_image_format pixel_format;
} XC_input_img;

typedef struct {
    XC_detect_box *boxes;
    XC_u32 count;
} XC_detect_result;

XC_s32 XC_image_infer_init(const XC_image_infer_config *config,
    XC_image_infer_handle **handle);

XC_s32 XC_image_infer_detect(XC_image_infer_handle *handle,
    const XC_input_img *input_img, XC_detect_result *result);

XC_void XC_image_infer_result_deinit(XC_detect_result *result);

XC_void XC_image_infer_destroy(XC_image_infer_handle *handle);

#ifdef __cplusplus
}
#endif

#endif
