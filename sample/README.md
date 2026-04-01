# test_image - NPU 单张图片推理示例程序

## 简介

`test_image` 是一个基于 `xmedia_cl` 底层接口的单张图片推理示例程序，用于：

- 读取一张 JPEG 图片
- 使用 `.xmm` 模型执行 NPU 推理
- 将图片转换为模型输入需要的 RGB CHW 格式
- 解析目标检测输出并打印候选框坐标
- 输出较详细的调试日志，便于分析 tensor shape、量化参数、padding 和框坐标映射问题

该程序适合用于：

- 快速验证模型是否能正常加载和运行
- 检查输入输出 tensor 布局
- 调试目标检测后处理逻辑
- 排查框坐标不准、tensor 对齐/stride 读取错误等问题

## 编译

```bash
# 进入SDK目录，设置环境变量
source env.sh

# 进入test_image目录
cd sample/npu/test_image

# 编译
make

# 清理
make clean
```

## 使用方法

### 基本用法

```bash
./test_image <image_path>
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `image_path` | 输入 JPEG 图片路径 |

### 示例

```bash
# 使用绝对路径
./test_image /mnt/sdcard/data/test.jpg

# 使用当前目录图片
./test_image ./test.jpg
```

## 输入图片格式

- 格式：JPEG (`.jpg` / `.jpeg`)
- 推理输入分辨率：`640x360`（由代码中的 `IMAGE_WIDTH/IMAGE_HEIGHT` 定义）
- 程序会先将 JPEG 解码，再缩放到 `640x360`，最后转成 RGB CHW 输入 tensor

### 建议

- 为了减少坐标映射误差，建议直接使用 `640x360` 的图片做测试
- 如果原图不是 `640x360`，程序会缩放后再推理，最终输出坐标对应的是模型输入坐标系
- 当前代码主要针对 `person` 类（`class_id = 0`）打印候选框

### 将其他格式转换为 JPEG

如果原图不是 JPEG，可先用 FFmpeg 转换：

```bash
# PNG 转 JPG
ffmpeg -i input.png -q:v 2 output.jpg

# 视频抽帧
ffmpeg -i video.mp4 -vframes 1 -q:v 2 frame.jpg
```

## 输出说明

程序运行时会输出以下几类信息：

1. 系统/CL 运行时初始化日志
2. 模型输入输出 tensor 信息
3. JPEG 解码与 RGB 数据摘要
4. 推理耗时
5. 检测候选框与 NMS 结果
6. 调试日志（用于排查坐标不准问题）

### 典型输出内容

```
=== NPU Image Inference Test (xmm mode) ===
Image: ./test.jpg (640x360)
Model: /mnt/sdcard/data/neuron_network.xmm

[Step 1] Initialize system...
System initialized.

[Step 13] Get output tensor info (2nd call)...
Output tensor metadata ready.

========================================
Output Tensor Layout
========================================
Tensor count: 6

Tensor #0:
  Size: 294912 bytes
  Shape dims: [1, 64, 48, 80, 0, 0, 0, 0]
  Quant: scale=0.117279 zp=75
  Physical width(from size): 96

...

========================================
Candidate Boxes
========================================

Level 0: stride=8, reg_tensor=0, cls_tensor=1, ...
raw_candidate[0] grid=(23,8) center=(188.00,68.00) ...

Person class only (class_id=0)
Found 9 raw candidate boxes above threshold 0.45
After class-wise NMS (IoU <= 0.45): 8 boxes
[000] cls=0 score=0.8502 box=(12.5, 0.0, 170.6, 359.0) stride=32 level=2
```

### 调试日志说明

当前版本加入了额外调试信息，重点用于分析后处理问题：

- `Input Tensor Layout` / `Output Tensor Layout`
  - 查看 tensor `size`、`shape dims`、量化参数、物理宽度
- `Physical width(from size)`
  - 用于判断 output tensor 是否带有硬件行对齐 padding
- `sample[...]`
  - 在固定 grid 点打印 cls/reg 原始值，方便检查 tensor 索引是否正确
- `raw_candidate[...]`
  - 打印候选框从 grid 点到最终 box 的解码过程
- `notice: level X grid height maps to ...`
  - 用于提示当前 Y 方向是否存在 384/360 这种坐标映射差异

## 模型文件

当前程序使用 `.xmm` 模型文件，默认路径为：

```c
#define MODEL_PATH "/mnt/sdcard/data/neuron_network.xmm"
```

请确保板端该路径存在对应模型文件。

## 依赖

- xmedia_cl（NPU/CL 运行时接口）
- xmedia_sys
- xmedia_mmz
- FFmpeg 相关库：
  - libavformat
  - libavcodec
  - libswscale
  - libavutil
- libm（数学库）
- libpthread（线程库）
- libstdc++（C++标准库）

## 与demo_ai的区别

| 特性 | demo_ai | test_image |
|------|---------|------------|
| 复杂度 | 完整系统 | 简化版/调试版 |
| 输入 | 视频流/图片/YUV | 单张 JPEG |
| 模型接口 | 上层业务封装 | xmedia_cl 直接调用 |
| 输出 | RTSP/视频/图片/业务结果 | 检测坐标 + 调试日志 |
| 功能 | 多种AI检测 | 单模型单图推理与后处理分析 |

## 常见问题

### Q: 编译报错 "cannot find -lavformat"
A: 当前版本需要 FFmpeg 解码 JPEG，请确认交叉编译环境中已经正确提供 FFmpeg 相关库。

### Q: 检测不到目标
A: 检查以下几点：

- 输入是否为 `.jpg` / `.jpeg`
- 图片内容是否与模型场景匹配
- 模型文件路径是否正确
- 图片是否被缩放后造成目标过小

### Q: 模型加载失败
A: 确保模型文件存在且路径正确

### Q: 输出 tensor 的 size 和 shape 对不上，是不是程序有问题？
A: 不一定。很多 NPU 输出 tensor 在内存里会带有硬件对齐 padding，逻辑 shape 里的宽度与实际物理宽度可能不同。当前代码已经增加 `Physical width(from size)` 日志用于分析这一问题。

### Q: 框坐标还是有偏差怎么办？
A: 先检查日志中的以下字段：

- `Physical width(from size)`
- `sample[...]`
- `raw_candidate[...]`
- `notice: level X grid height maps to ...`

这些日志可以帮助判断问题出在：

- tensor 索引是否读错
- output 是否带 padding
- stride 是否推导正确
- Y 方向是否存在 384 -> 360 的映射差异

## 技术细节

### 核心API调用流程

```
xmedia_sys_init()           # 初始化系统
    ↓
xmedia_cl_init()            # 初始化 CL 运行时
    ↓
xmedia_cl_create_context()  # 创建上下文
    ↓
xmedia_cl_graph_loadmodel_from_file_withmem()  # 加载 .xmm 模型
    ↓
xmedia_cl_graph_get_input()/get_output()       # 获取 tensor 信息
    ↓
xmedia_cl_graph_set_inout()                     # 绑定输入输出
    ↓
xmedia_cl_graph_process()                       # 执行推理
    ↓
print_candidate_boxes()                        # 后处理并打印检测框
```

### 内存管理

- 使用 MMZ (Memory Management Zone) 分配物理内存
- 通过 xmedia_mmz_alloc/map/unmap/free 管理
- 输入输出 tensor 通过一整块 MMZ 内存顺序切分，并按 16 字节对齐

### 后处理说明

当前后处理逻辑采用：

- 3 个检测 level
- 回归头通道数 `64`（`DFL_BINS * 4`）
- 分类头通道数 `80`
- 只输出 `person(class_id=0)`
- DFL 解码 + class-wise NMS

另外，当前代码已经针对以下问题做了专门处理：

1. **output tensor padding**
   - 不再直接用 `dims[3]` 作为真实行跨度
   - 改为根据 `tensor->size` 反推物理宽度

2. **Y 方向 384/360 映射差异**
   - 当前按调试结果采用“顶部对齐、Y 不缩放”的策略
   - 最终仍会 clamp 到 `0 ~ 359`

## 许可

Copyright (c) XMEDIA. All rights reserved.
