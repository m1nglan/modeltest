# ESP32-S3 螺丝/垫圈二分类模型测试项目

## 概述

在 ESP32-S3 (N16R8) 上部署 INT8 量化 MobileNetV2 风格二分类模型，对 `t1.jpg`（垫圈/washer）和 `t2.jpg`（螺丝/screw）进行实时分类推理。

**硬件平台**: ESP32-S3 (16MB Flash + 8MB Octal PSRAM)  
**框架**: ESP-IDF v6.0.1 + ESP-DL (managed_components)  
**模型**: MobileNetV2 风格 INT8 量化二分类器, NHWC `[1x224x224x3]`, exponent=-5  
**量化工具**: ESP-PPQ (PPQ-based)

---

## 项目文件结构

### 根目录

| 文件 | 说明 |
|------|------|
| `CMakeLists.txt` | ESP-IDF 顶层 CMake 项目配置 |
| `partitions.csv` | 自定义分区表: factory 6400K, 模型内嵌在 app rodata |
| `sdkconfig.defaults` | 默认配置: 16MB Flash, 8MB Octal PSRAM, WDT 关闭, INT_WDT 30s |
| `verify_model.py` | PC 端预处理 + ONNX Runtime 验证脚本（与 ESP32 驱动同公式） |
| `gen_test_data.py` | 用 PC 预处理公式生成 INT8 输入数据，输出 `main/test_inputs.h` |
| `serial_capture.py` | 串口捕获 ESP32 回传 raw 图片数据的工具脚本（未启用） |
| `dependencies.lock` | ESP-IDF 组件依赖锁定文件 |

### `main/` 目录

| 文件 | 说明 |
|------|------|
| `main.cpp` | 主入口: init → run_diagnostic → infer(t1.jpg) → infer(t2.jpg) → infer_from_preprocessed(t1/t2) |
| `CMakeLists.txt` | 组件构建配置: 编译源文件 + 嵌入 model.espdl / t1.jpg / t2.jpg 到 rodata |
| `idf_component.yml` | 组件依赖: espressif/esp-dl: "*", espressif/esp_new_jpeg |
| `model.espdl` | ESP-DL FlatBuffers 格式的 INT8 量化模型（NHWC） |
| `model.info` | PPQ 导出的模型信息: 算子列表、张量形状、exponent 参数 |
| `model.json` | PPQ 导出的量化配置 JSON: 每层量化策略（per-tensor, symmetric, power-of-2） |
| `t1.jpg` | 测试图 1: 垫圈 (washer), class 0, 640×480 |
| `t2.jpg` | 测试图 2: 螺丝 (screw), class 1, 640×480 |
| `test_inputs.h` | 由 `gen_test_data.py` 生成, PC 预处理后的 INT8 输入数据, 用于跳过预处理直接测试模型 |
| `data.log` | 多次烧录测试的串口日志归档 |

### `main/driver/` 目录

| 文件 | 说明 |
|------|------|
| `classifier_driver.hpp` | 驱动头文件: `ClassifierDriver` 类声明, `classification_result_t` 结构体 |
| `classifier_driver.cpp` | 驱动实现: 模型加载, JPEG 解码, ImagePreprocessor (NHWC), 手动 HWC/NCHW 量化, 输出反量化+softmax+argmax |

### `managed_components/` 目录（ESP-IDF 自动管理）

| 组件 | 说明 |
|------|------|
| `espressif__esp-dl/` | ESP-DL 推理框架 (commit 77a8a624), 含 Conv/Pool/Add/Clip 等算子 |
| `espressif__dl_fft/` | DL FFT 数学库依赖 |
| `espressif__esp_new_jpeg/` | 硬件 JPEG 解码库 |

---

## 当前状态

### ✅ 已完成

1. **项目脚手架**: CMake + 分区表 + sdkconfig 配置完成
2. **模型加载**: 从 rodata 加载 `model.espdl`, ESP-DL 自动检测 NHWC 格式
3. **推理流水线**: JPEG 解码 → resize/crop → 归一化+量化 → Conv 推理 → 输出反量化 + softmax + argmax
4. **PC 端验证**: `verify_model.py` 用 ONNX Runtime 跑 INT8 ONNX 模型, 结果正确
5. **跳过预处理测试**: `infer_from_preprocessed()` 直接用 PC 预处理好的 INT8 数据灌入模型, 排除预处理差异

### ❌ 未解决

#### 核心问题: ESP-DL INT8 推理结果不正确

即使使用 PC 预处理好的完全正确的 INT8 输入数据, ESP-DL 推理结果仍有错误:

| | PC (FP32 ONNX) | ESP32 (INT8) |
|---|---|---|
| t1 (垫圈, class 0) | `[8.25, -9.50]` → class 0 ✅ | `[7.875, -9.125]` → class 0 ✅ |
| t2 (螺丝, class 1) | `[-12.72, 12.20]` → class 1 ✅ | `[7.000, -8.125]` → class 0 ❌ |

t2 的 class 0/1 符号与 PC 完全相反。

#### 其它已知问题

1. **ESP-DL 自检失败**: 模型内嵌测试数据对比 PPQ 仿真值为 `16`, ESP-DL 实际计算为 `106`（4 倍差异）, 说明 PPQ 仿真与 ESP-DL 运行时存在计算不一致
2. **JPEG 解码偏差**: `sw_decode_jpeg` 输出的 raw 像素值 (134) 与 PC 的 PIL 解码 (157) 不同, 导致 JPEG 路径的 INT8 输入进一步偏差 (6 vs 18)
3. **PPQ 量化噪声**: 早期版本 3 层 Conv 噪声 >100%, 优化后降至 ~0.028%（classifier 层）, 量化精度已够

---

## 文件职责速查

### 训练/量化端（需单独排查）

- **模型训练**: MobileNetV2 风格二分类 PyTorch 训练脚本（不在本项目）
- **PPQ 量化**: ESP-PPQ 量化脚本, 导出 `model.espdl` / `model.info` / `model.json`
- **`model.espdl`**: 需确认导出时启用 `export_test_values=True` 以启用自检
- **`model.info`**: 包含所有 53 层 Conv 的 group、activation、kernel_shape 等参数

### 推理端（本项目）

| 文件 | 调试状态 | 备注 |
|------|---------|------|
| `classifier_driver.cpp` | 含完整调试打印 | raw 像素, INT8 输入值, 输出 logit, 三种诊断模式 |
| `main.cpp` | 分步骤执行 | JPG 推理 + PC 预处理数据推理 |
| `verify_model.py` | 可用 | 需确保 `.onnx` 与 `.espdl` 权重一致 |
| `gen_test_data.py` | 可用 | 生成 `test_inputs.h` 供 ESP32 跳过预处理 |

---

## 排查计划

### 第 1 步: 确认 PPQ 导出正确性
- [ ] PPQ 导出 `model.espdl` 时启用 `export_test_values=True`
- [ ] 在 ESP32 上观察自检输出, 定位哪一层开始数值不一致（输入层? 第1层Conv? 第N层?）

### 第 2 步: 确认 ONNX INT8 与 FP32 一致性
- [ ] 用 PPQ 导出 INT8 ONNX 模型, 在 PC 上 ONNX Runtime 跑相同的输入
- [ ] 确认 INT8 ONNX 输出与 FP32 ONNX 一致

### 第 3 步: 逐层对比 ESP-DL vs PPQ 仿真
- [ ] 用 PPQ 导出每层的中间张量测试数据
- [ ] ESP-DL 逐层对比, 找到第一个偏差的算子

### 第 4 步: 修复或绕过
- [ ] 定位到具体算子（如 Conv/Add/Clip）的 INT8 实现差异后, 修复驱动或调整 PPQ 量化策略
- [ ] 或考虑使用 NCHW 格式模型 + 手动量化路径绕过 ImagePreprocessor

---

## 使用方式

```bash
# 首次构建
idf.py set-target esp32s3
idf.py build

# 烧录（可能需要按住 BOOT + 按 ENTER）
idf.py -p COM22 flash

# 监视串口
idf.py -p COM22 monitor

# PC 端验证
python verify_model.py

# 生成 PC 预处理测试数据（重新编译后烧录）
python gen_test_data.py
idf.py build -p COM22 flash monitor
```
