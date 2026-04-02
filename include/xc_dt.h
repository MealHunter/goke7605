#ifndef __XC_DT_H__
#define __XC_DT_H__

#include "xmedia_cl.h"
#include "xc_common_datatype.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef struct XC_image_infer_handle XC_image_infer_handle;

typedef struct {
    const char *model_path;
    XC_U32 image_width;
    XC_U32 image_height;
    XC_FLOAT score_thresh;
    XC_FLOAT nms_iou_thresh;
    XC_U32 person_class_id;
} XC_image_infer_config;

typedef struct {
    XC_S32 class_id;
    XC_FLOAT score;
    XC_FLOAT x1;
    XC_FLOAT y1;
    XC_FLOAT x2;
    XC_FLOAT y2;
    XC_U32 stride;
    XC_U32 level;
} XC_detect_box;

typedef enum {
    XC_IMAGE_FORMAT_RGB888 = 0,
    XC_IMAGE_FORMAT_BGR888 = 1,
} XC_image_format;

typedef struct {
    XC_U64 phy_addr;
    XC_U32 width;
    XC_U32 height;
    XC_U32 stride;
    XC_image_format pixel_format;
} XC_input_img;

typedef struct {
    XC_detect_box *boxes;
    XC_U32 count;
} XC_detect_result;

XC_S32 XC_image_infer_init(const XC_image_infer_config *config, XC_image_infer_handle **handle);

XC_S32 XC_image_infer_detect(XC_image_infer_handle *handle, const XC_input_img *input_img, XC_detect_result *result);

void XC_image_infer_result_deinit(XC_detect_result *result);

void XC_image_infer_destroy(XC_image_infer_handle *handle);

#ifdef __cplusplus
}
#endif

#endif
