# xm image infer library

基于 `sample/test_image.c` 拆出一层可复用静态库结构：

```text
.
├── include/
│   └── xm_image_infer.h      # 对外接口
├── src/
│   └── xm_image_infer.c      # 库实现（解码 + 推理 + 后处理）
├── examples/
│   └── test_image_demo.c     # 调用示例
├── lib/                      # make 后生成 libxm_image_infer.a
├── bin/                      # make 后生成示例程序
├── build/                    # make 后生成中间产物
└── Makefile
```

## 当前对外接口

- `xm_image_infer_init()`：初始化模型和运行时
- `xm_image_infer_detect()`：输入 JPEG，执行推理并返回检测框
- `xm_image_infer_result_deinit()`：释放结果内存
- `xm_image_infer_destroy()`：释放推理句柄

## 构建

在 SDK 环境下执行：

```bash
make
```

生成物：

- `lib/libxm_image_infer.a`
- `bin/test_image_demo`

## 后续建议

如果你接下来要继续“封装接口”，建议下一步再拆成：

1. `src/xm_image_decoder.c`：图片解码/预处理
2. `src/xm_image_runtime.c`：xmedia runtime 生命周期
3. `src/xm_image_postprocess.c`：YOLO 后处理

这样后续维护会比单文件库更方便。
