# xm image infer library

基于 `sample/test_image.c` 拆出一层可复用静态库结构，供固件应用层直接传图像帧调用：

```text
.
├── include/
│   └── xm_image_infer.h      # 对外接口
├── src/
│   └── XC_image_infer.c      # 库实现（输入帧搬运 + 推理 + 后处理）
├── examples/
│   └── test_image_demo.c     # 调用示例
├── lib/                      # make 后生成 libXC_image_infer.a
├── bin/                      # make 后生成示例程序
├── build/                    # make 后生成中间产物
└── Makefile
```

## 当前对外接口

- `XC_image_infer_init()`：初始化模型和运行时
- `XC_image_infer_detect()`：输入图像帧信息，执行推理并返回检测框
- `XC_image_infer_result_deinit()`：释放结果内存
- `XC_image_infer_destroy()`：释放推理句柄

## 输入帧结构

```c
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
```

当前 `XC_image_infer_detect()` 约束：

- 输入由应用层传入物理地址
- 当前支持 `RGB888` / `BGR888` 打包格式
- 当前要求输入帧宽高与模型输入宽高一致
- 检测接口内部会自行 map/unmap 物理地址并完成 CHW 数据搬运

## 示例

```c
XC_input_img input_img = {
    .phy_addr = frame_phy_addr,
    .width = 640,
    .height = 360,
    .stride = 640 * 3,
    .pixel_format = XC_IMAGE_FORMAT_RGB888,
};

ret = XC_image_infer_init(&config, &handle);
ret = XC_image_infer_detect(handle, &input_img, &result);
```

## 构建

在 SDK 环境下执行：

```bash
make
```

生成物：

- `lib/libXC_image_infer.a`
- `bin/test_image_demo`

## 后续建议

如果你接下来要继续“封装接口”，建议下一步再拆成：

1. `src/XC_image_decoder.c`：图片解码/预处理
2. `src/XC_image_runtime.c`：xmedia runtime 生命周期
3. `src/XC_image_postprocess.c`：YOLO 后处理

这样后续维护会比单文件库更方便。
