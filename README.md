# XC image infer library

这是一个给固件应用层直接调用的 **NPU 单帧检测静态库**。

它做的事情可以简单理解为：

1. 先加载模型，初始化 NPU 运行环境
2. 应用层把一帧图像的物理地址传进来
3. 库内部把图像整理成模型输入格式
4. 执行推理
5. 把检测框结果返回给应用层

如果你是第一次接触这个项目，可以先记住一句话：

> **这个库不是拿“图片路径”做检测，而是拿“图像帧信息”做检测。**

---

## 1. 这个项目适合什么场景

适合下面这种板端/固件场景：

- 你已经拿到一帧图像
- 这帧图像在物理内存里
- 你想直接把这帧图像送给 NPU
- 你希望拿到检测框坐标

它不适合下面这种场景：

- 直接传 JPG/PNG 文件路径
- 在 PC 上独立运行、没有板端 SDK 环境
- 希望库内部帮你做完整的图片解码

当前版本更偏向“**给固件应用层调用的底层检测封装**”。

---

## 2. 项目目录说明

```text
.
├── include/
│   ├── XC_image_infer.h         # 本库对外头文件
│   ├── XC_common_datatype.h     # XC 基础类型定义
│   ├── xmedia_cl.h              # 依赖的 SDK 头文件
│   └── xmedia_cl_common.h       # 依赖的 SDK 头文件
├── src/
│   └── XC_image_infer.c         # 核心实现：输入搬运 + 推理 + 后处理
├── examples/
│   └── test_image_demo.c        # 最小调用示例
├── lib/                         # make 后生成静态库
├── bin/                         # make 后生成示例程序
├── build/                       # make 后生成中间文件
├── Makefile                     # 构建脚本
└── README.md
```

如果你是新手，最建议先看这 3 个文件：

1. `include/XC_image_infer.h`
2. `examples/test_image_demo.c`
3. `src/XC_image_infer.c`

---

## 3. 这个库最终提供什么能力

这个库主要提供 4 个对外接口：

- `XC_image_infer_init()`
- `XC_image_infer_detect()`
- `XC_image_infer_result_deinit()`
- `XC_image_infer_destroy()`

你可以把它理解成两步主流程：

### 第一步：初始化

调用 `XC_image_infer_init()`：

- 初始化系统
- 初始化 NPU/CL 运行环境
- 加载模型文件
- 准备输入输出 tensor
- 分配推理所需内存

### 第二步：检测

调用 `XC_image_infer_detect()`：

- 输入一帧图像信息
- 从物理地址映射出图像数据
- 转成模型需要的 CHW 输入格式
- 执行推理
- 解析输出 tensor
- 返回检测框

### 最后：释放资源

- `XC_image_infer_result_deinit()`：释放本次检测结果
- `XC_image_infer_destroy()`：释放初始化时创建的句柄和底层资源

---

## 4. 头文件和命名说明

当前工程里有一个容易让新手困惑的点，需要提前说明：

- **头文件文件名**：`include/XC_image_infer.h`
- **对外函数/类型前缀**：`XC_`

也就是说，你在代码里会写：

```c
#include "XC_image_infer.h"
```

然后调用：

```c
XC_image_infer_init(...)
XC_image_infer_detect(...)
```

---

## 5. 构建方式

在 SDK 环境下执行：

```bash
make
```

生成物：

- `lib/libXC_image_infer.a`
- `bin/test_image_demo`

### 说明

这个工程依赖板端 SDK，所以通常不能在普通 PC 环境直接编过。

如果你编译时报错，先确认：

- 已经进入正确的 SDK 环境
- `env.sh` 或等价环境已经 source
- `xmedia_cl.h` 等依赖头文件可见
- 对应库文件可链接

---

## 6. 对外数据结构说明

这一部分最重要，新手一定要看。

### 6.1 初始化配置：`XC_image_infer_config`

```c
typedef struct {
    const XC_char *model_path;
    XC_U32 image_width;
    XC_U32 image_height;
    XC_FLOAT score_thresh;
    XC_FLOAT nms_iou_thresh;
    XC_U32 person_class_id;
} XC_image_infer_config;
```

字段说明：

- `model_path`
  - 模型文件路径
  - 例如：`/mnt/sdcard/data/neuron_network.xmm`

- `image_width`
  - 模型输入宽度
  - 例如：`640`

- `image_height`
  - 模型输入高度
  - 例如：`360`

- `score_thresh`
  - 置信度阈值
  - 分数低于这个值的框会被过滤掉

- `nms_iou_thresh`
  - NMS 阈值
  - 用来去掉大量重叠框

- `person_class_id`
  - 当前关注的类别 ID
  - 例如人通常是 `0`

---

### 6.2 输入图像结构：`XC_input_img`

```c
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
```

这个结构表示“应用层要送进来的一帧图像”。

字段说明：

- `phy_addr`
  - 图像帧的物理地址
  - 库内部会基于这个地址去 map 图像数据

- `width`
  - 图像宽度

- `height`
  - 图像高度

- `stride`
  - 每行实际占用字节数
  - 注意：**stride 不一定等于 width * 3**
  - 如果底层做了对齐，stride 可能更大

- `pixel_format`
  - 当前支持：
    - `XC_IMAGE_FORMAT_RGB888`
    - `XC_IMAGE_FORMAT_BGR888`

---

### 6.3 检测框结构：`XC_detect_box`

```c
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
```

字段说明：

- `class_id`：类别 ID
- `score`：该框置信度
- `x1, y1`：左上角坐标
- `x2, y2`：右下角坐标
- `stride`：该框来自哪个检测层对应的 stride
- `level`：该框来自哪个检测层

---

### 6.4 检测结果结构：`XC_detect_result`

```c
typedef struct {
    XC_detect_box *boxes;
    XC_U32 count;
} XC_detect_result;
```

字段说明：

- `boxes`：检测框数组
- `count`：检测框数量

注意：

> `boxes` 里的内存是库分配的，所以用完后要调用 `XC_image_infer_result_deinit()` 释放。

---

## 7. 最常见的调用流程

推荐你按下面顺序使用：

```text
1. 组织配置参数
2. 调用 XC_image_infer_init()
3. 准备一帧 XC_input_img
4. 调用 XC_image_infer_detect()
5. 遍历结果框
6. 调用 XC_image_infer_result_deinit()
7. 调用 XC_image_infer_destroy()
```

如果你是第一次接入，先照着这个顺序走，不要跳步骤。

---

## 8. 最小调用示例

下面是一个最容易理解的示例：

```c
#include "XC_image_infer.h"

#define MODEL_PATH "/mnt/sdcard/data/neuron_network.xmm"

int main(void)
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
    XC_detect_result result = {0};

    XC_input_img input_img = {
        .phy_addr = frame_phy_addr,
        .width = 640,
        .height = 360,
        .stride = 640 * 3,
        .pixel_format = XC_IMAGE_FORMAT_RGB888,
    };

    if (XC_image_infer_init(&config, &handle) != 0) {
        return -1;
    }

    if (XC_image_infer_detect(handle, &input_img, &result) != 0) {
        XC_image_infer_destroy(handle);
        return -1;
    }

    for (XC_U32 i = 0; i < result.count; i++) {
        XC_detect_box *box = &result.boxes[i];
        printf("box[%u]: cls=%d score=%.4f (%.1f, %.1f, %.1f, %.1f)\n",
            i, box->class_id, box->score,
            box->x1, box->y1, box->x2, box->y2);
    }

    XC_image_infer_result_deinit(&result);
    XC_image_infer_destroy(handle);
    return 0;
}
```

---

## 9. 示例程序怎么运行

项目里自带了一个最小示例：

- `examples/test_image_demo.c`

它的命令行参数是：

```bash
./test_image_demo <phy_addr_hex> <width> <height> <stride> <pixel_format>
```

参数说明：

- `phy_addr_hex`
  - 图像帧物理地址，通常十六进制传入

- `width`
  - 图像宽

- `height`
  - 图像高

- `stride`
  - 每行字节数

- `pixel_format`
  - `0` 表示 `XC_IMAGE_FORMAT_RGB888`
  - `1` 表示 `XC_IMAGE_FORMAT_BGR888`

例如：

```bash
./test_image_demo 0x84000000 640 360 1920 0
```

这里的 `1920` 就是 `640 * 3`。

---

## 10. 库内部大概做了什么

如果你不想只会“调用”，也想知道内部逻辑，可以看这里。

### `XC_image_infer_init()` 内部做的事

- 初始化系统
- 初始化 CL 运行环境
- 查询设备
- 创建设备上下文
- 加载模型
- 获取输入输出 tensor 信息
- 分配输入输出所需内存

### `XC_image_infer_detect()` 内部做的事

- 检查输入参数是否合法
- 用物理地址 map 出图像数据
- 按 RGB/BGR 格式搬运数据
- 转成 CHW 布局
- 绑定输入输出 tensor
- 执行图推理
- 解码检测结果
- 做 NMS
- 返回框数组

### `XC_image_infer_result_deinit()` 内部做的事

- 释放检测框数组

### `XC_image_infer_destroy()` 内部做的事

- 卸载 graph
- 释放输入输出内存
- 释放 work/weight 内存
- 释放 context / device
- 退出运行环境

---

## 11. 当前版本的限制

这一段很重要，能帮你少踩很多坑。

### 11.1 当前不支持图片路径输入

当前版本已经不是“传 JPG 路径检测”的方案了。

现在必须传：

- 物理地址
- 宽高
- stride
- 像素格式

---

### 11.2 当前只支持两种像素格式

只支持：

- `XC_IMAGE_FORMAT_RGB888`
- `XC_IMAGE_FORMAT_BGR888`

暂不支持：

- NV12
- NV21
- YUV420
- 灰度图

如果你的上游给的是 NV12/NV21，需要你后续再扩展，或者让我继续帮你加。

---

### 11.3 当前要求输入尺寸和模型尺寸一致

如果模型要求 `640x360`，那你传入的图像也必须是 `640x360`。

当前版本还没有做：

- resize
- letterbox
- padding 补边

所以尺寸不一致时，检测会直接失败。

---

### 11.4 当前更偏向单类别检测

配置里有：

- `person_class_id`

这说明当前实现更偏向“只关心某一个类别”，而不是完整地把所有类别都输出给上层。

---

## 12. 新手最容易踩的坑

### Q1：为什么 detect 直接返回失败？

优先检查：

- `handle` 是否初始化成功
- `phy_addr` 是否为 0
- `width/height/stride` 是否正确
- `pixel_format` 是否是 0 或 1
- 输入图像尺寸是否和模型输入尺寸一致

---

### Q2：为什么没有检测框？

可能原因：

- 图像内容里本来就没有目标
- `score_thresh` 设置太高
- `person_class_id` 不对
- 输入颜色格式传错了（RGB/BGR 搞反）
- 输入地址对应的数据不是真正的图像帧

---

### Q3：为什么框坐标看起来不对？

可能原因：

- 输入帧宽高不对
- stride 写错
- RGB / BGR 传反
- 上游传入的图像和你以为的图像格式不一致

---

### Q4：为什么程序崩溃？

常见原因：

- 传入了错误的物理地址
- 物理地址不可 map
- 输入 stride 不合法
- 没有在正确 SDK 环境下运行

---

### Q5：为什么结果用完后还要手动释放？

因为结果框数组是库内部动态申请的。

所以你必须在用完后调用：

```c
XC_image_infer_result_deinit(&result);
```

否则可能造成内存泄漏。

---

## 13. 推荐的阅读顺序

如果你是第一次接触这个库，推荐按下面顺序读：

1. 先看本 README
2. 再看 `include/XC_image_infer.h`
3. 再看 `examples/test_image_demo.c`
4. 最后看 `src/XC_image_infer.c`

这样会最容易建立完整理解。

---

## 14. 后续可以继续扩展什么

如果你接下来还要继续做功能增强，通常会优先做下面几项：

1. 支持 NV12 / NV21 输入
2. 支持输入尺寸和模型尺寸不一致时自动缩放
3. 支持多类别输出
4. 把输入处理、运行时、后处理拆成多个 `.c` 文件
5. 增加更完整的错误码

---

## 15. 一句话总结

这个库的核心目标就是：

> **让固件应用层只需要准备一帧图像和一个模型，就能直接拿到检测框结果。**

如果你接下来想要，我还可以继续帮你把 README 再补成：

- “接口逐行解释版”
- “给应用层同事的接入文档版”
- “问题排查手册版”
