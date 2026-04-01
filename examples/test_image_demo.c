#include <stdio.h>

#include "../include/xm_image_infer.h"

#define MODEL_PATH "/mnt/sdcard/data/neuron_network.xmm"

int main(int argc, char *argv[])
{
    xm_image_infer_config config = {
        .model_path = MODEL_PATH,
        .image_width = 640,
        .image_height = 360,
        .score_thresh = 0.45f,
        .nms_iou_thresh = 0.45f,
        .person_class_id = 0,
    };
    xm_image_infer_handle *handle = NULL;
    xm_detect_result result = {0};
    xmedia_s32 ret;
    xmedia_u32 i;

    if (argc < 2) {
        printf("Usage: %s <image_path>\n", argv[0]);
        return -1;
    }

    ret = xm_image_infer_init(&config, &handle);
    if (ret != 0) {
        printf("xm_image_infer_init failed: %d\n", ret);
        return ret;
    }

    ret = xm_image_infer_detect(handle, argv[1], &result);
    if (ret != 0) {
        printf("xm_image_infer_detect failed: %d\n", ret);
        xm_image_infer_destroy(handle);
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

    xm_image_infer_result_deinit(&result);
    xm_image_infer_destroy(handle);
    return 0;
}
