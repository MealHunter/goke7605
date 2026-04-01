#include <stdio.h>
#include <stdlib.h>

#include "../include/xm_image_infer.h"

#define MODEL_PATH "/mnt/sdcard/data/neuron_network.xmm"

int main(int argc, char *argv[])
{
    XC_image_infer_config config = {
        .model_path = MODEL_PATH,
        .image_width = 640,
        .image_height = 360,
        .score_thresh = 0.45f,
        .nms_iou_thresh = 0.45f,
        .person_class_id = 0,
    };
    XC_image_infer_handle *handle = NULL;
    XC_input_img input_img;
    XC_detect_result result = {0};
    XC_s32 ret;
    XC_u32 i;

    if (argc < 6) {
        printf("Usage: %s <phy_addr_hex> <width> <height> <stride> <pixel_format>\n", argv[0]);
        printf("pixel_format: 0=RGB888, 1=BGR888\n");
        return -1;
    }

    input_img.phy_addr = (XC_u64)strtoull(argv[1], NULL, 0);
    input_img.width = (XC_u32)strtoul(argv[2], NULL, 0);
    input_img.height = (XC_u32)strtoul(argv[3], NULL, 0);
    input_img.stride = (XC_u32)strtoul(argv[4], NULL, 0);
    input_img.pixel_format = (XC_image_format)strtoul(argv[5], NULL, 0);

    ret = XC_image_infer_init(&config, &handle);
    if (ret != 0) {
        printf("XC_image_infer_init failed: %d\n", ret);
        return ret;
    }

    ret = XC_image_infer_detect(handle, &input_img, &result);
    if (ret != 0) {
        printf("XC_image_infer_detect failed: %d\n", ret);
        XC_image_infer_destroy(handle);
        return ret;
    }

    printf("detect count: %u\n", result.count);
    for (i = 0; i < result.count; i++) {
        printf("[%03u] cls=%d score=%.4f box=(%.1f, %.1f, %.1f, %.1f) stride=%u level=%u\n",
            i,
            result.boxes[i].class_id,
            result.boxes[i].score,
            result.boxes[i].x1,
            result.boxes[i].y1,
            result.boxes[i].x2,
            result.boxes[i].y2,
            result.boxes[i].stride,
            result.boxes[i].level);
    }

    XC_image_infer_result_deinit(&result);
    XC_image_infer_destroy(handle);
    return 0;
}
