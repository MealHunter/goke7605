#ifndef XM_IMAGE_INFER_H
#define XM_IMAGE_INFER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "xmedia_cl.h"

typedef struct xm_image_infer_handle xm_image_infer_handle;

typedef struct {
    const xmedia_char *model_path;
    xmedia_u32 image_width;
    xmedia_u32 image_height;
    xmedia_float score_thresh;
    xmedia_float nms_iou_thresh;
    xmedia_u32 person_class_id;
} xm_image_infer_config;

typedef struct {
    xmedia_s32 class_id;
    xmedia_float score;
    xmedia_float x1;
    xmedia_float y1;
    xmedia_float x2;
    xmedia_float y2;
    xmedia_u32 stride;
    xmedia_u32 level;
} xm_detect_box;

typedef enum {
    XM_IMAGE_FORMAT_RGB888 = 0,
    XM_IMAGE_FORMAT_BGR888 = 1,
} xm_image_format;

typedef struct {
    xmedia_u64 phy_addr;
    xmedia_u32 width;
    xmedia_u32 height;
    xmedia_u32 stride;
    xm_image_format pixel_format;
} xm_input_img;

typedef struct {
    xm_detect_box *boxes;
    xmedia_u32 count;
} xm_detect_result;

xmedia_s32 xm_image_infer_init(const xm_image_infer_config *config,
    xm_image_infer_handle **handle);

xmedia_s32 xm_image_infer_detect(xm_image_infer_handle *handle,
    const xm_input_img *input_img, xm_detect_result *result);

xmedia_void xm_image_infer_result_deinit(xm_detect_result *result);

xmedia_void xm_image_infer_destroy(xm_image_infer_handle *handle);

#ifdef __cplusplus
}
#endif

#endif
