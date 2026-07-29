# ClassifierDriver — ESP-DL 二分类驱动

通用 ESP-DL INT8 NHWC 二分类驱动，支持任意 2 类模型。
流水线：JPEG 解码 → ImagePreprocessor(resize+crop+量化) → ESP-DL 推理 → softmax

**硬件**: ESP32-S3 (N16R8: 16MB Flash + 8MB Octal PSRAM)  
**框架**: ESP-IDF v6.0.1 + ESP-DL (managed_components)  
**模型**: INT8 量化 NHWC `[1x224x224x3]` 二分类器  
**量化工具**: ESP-PPQ

---

## 📦 快速使用

### 1. 放置模型和图片

把以下文件放入 `main/` 目录：

```
main/
├── model.espdl      # ESP-PPQ 导出的模型文件（NHWC, INT8, [1x224x224x3]）
├── t1.jpg           # 测试图片 1
├── t2.jpg           # 测试图片 2
├── ...
└── driver/
    ├── classifier_driver.hpp
    └── classifier_driver.cpp
```

### 2. 配置 CMakeLists.txt

`main/CMakeLists.txt` 中注册源文件和要嵌入的二进制数据：

```cmake
idf_component_register(SRCS "main.cpp" "driver/classifier_driver.cpp"
                    INCLUDE_DIRS "." "driver"
                    REQUIRES espressif__esp-dl)

# 嵌入模型和数据文件到 rodata
set(embed_bin_files
    "model.espdl"
    "t1.jpg"
    "t2.jpg"
)
foreach(bin_file ${embed_bin_files})
    target_add_aligned_binary_data(${COMPONENT_LIB} "${CMAKE_CURRENT_SOURCE_DIR}/${bin_file}" BINARY)
endforeach()
```

### 3. 编写主程序

```cpp
#include "driver/classifier_driver.hpp"

// 链接器自动生成 _binary_*_start / _binary_*_end 符号
extern const uint8_t t1_jpg_start[] asm("_binary_t1_jpg_start");
extern const uint8_t t1_jpg_end[]   asm("_binary_t1_jpg_end");

extern "C" void app_main(void)
{
    ClassifierDriver driver;
    driver.init();

    size_t len = (size_t)(t1_jpg_end - t1_jpg_start);
    classification_result_t res = driver.infer(t1_jpg_start, len);
    // res.class_id → 0=screw, 1=washer
    // res.score    → 反量化后的 logit 值
    // res.probability → softmax 概率
    // res.label    → "screw" 或 "washer"
}
```

### 4. sdkconfig.defaults 配置

```ini
# Flash 大小（N16R8 = 16MB）
CONFIG_ESPTOOLPY_FLASHSIZE_16MB=y

# PSRAM（8MB Octal）
CONFIG_ESP32S3_SPIRAM_SUPPORT=y
CONFIG_SPIRAM_MODE_OCT=y
CONFIG_SPIRAM_TYPE_ESPPSRAM64=y
CONFIG_SPIRAM_SPEED_80M=y
CONFIG_SPIRAM_USE_CAPS_ALLOC=y

# 关掉任务看门狗（推理约 3s/张）
CONFIG_TASK_WDT=n
CONFIG_INT_WDT_TIMEOUT_MS=30000

# 自定义分区表（factory 6400K 起）
CONFIG_PARTITION_TABLE_CUSTOM=y
CONFIG_PARTITION_TABLE_CUSTOM_FILENAME="partitions.csv"
```

---

## 🧩 API 文档

### `classification_result_t`

```c
typedef struct {
    int class_id;        // 0=screw(螺丝), 1=washer(垫圈)
    float score;         // 反量化后的原始 logit 值
    float probability;   // Softmax 概率 [0, 1]
    const char *label;   // "screw" 或 "washer"
} classification_result_t;
```

### `ClassifierDriver`

| 方法 | 说明 |
|------|------|
| `ClassifierDriver()` | 构造函数，不执行初始化 |
| `~ClassifierDriver()` | 析构函数，释放模型和预处理器 |
| `esp_err_t init()` | 加载 `model.espdl`（从 rodata）+ 创建 ImagePreprocessor |
| `infer(jpg_data, len)` | JPEG → 解码 → 预处理 → 推理 → 返回分类结果 |
| `infer_from_preprocessed(int8_data)` | 直接喂 INT8 数据推理（跳过预处理，调试用） |

---

## 🔧 工作原理

### 预处理

ImagePreprocessor 内部自动完成：
1. **resize** — 短边缩放到 256，保持宽高比（BILINEAR）
2. **center crop** — 裁切到 224×224
3. **归一化** — `(pixel - mean) / std`，mean/std 为 [0,255] 范围
4. **量化** — `round(value / 2^exponent)`，exponent 从模型自动读取

归一化参数（ImageNet 标准）：

| 通道 | mean (×255) | std (×255) |
|------|------------|-----------|
| R    | 123.675    | 58.395    |
| G    | 116.28     | 57.12     |
| B    | 103.53     | 57.375    |

### 模型要求

| 属性 | 要求 |
|------|------|
| 格式 | ESP-DL FlatBuffers (`.espdl`) |
| 量化 | INT8 对称量化 (zero_point=0) |
| 布局 | NHWC `[1, 224, 224, 3]` |
| 输入 exponent | -6（对应 scale=0.015625） |
| 输出 | 2 个 INT8 logit，exponent=-4 |
| 类别 0 | screw（螺丝） |
| 类别 1 | washer（垫圈） |

---

## 📁 项目文件

| 文件 | 说明 |
|------|------|
| `main/model.espdl` | **模型文件** — ESP-PPQ 导出，放入 main/ 即可 |
| `main/*.jpg` | **测试图片** — 放入 main/，在 CMakeLists.txt 注册 |
| `main/driver/classifier_driver.hpp` | 驱动头文件 — API 声明 |
| `main/driver/classifier_driver.cpp` | 驱动实现 |
| `main/main.cpp` | 主程序示例 |
| `main/CMakeLists.txt` | 组件构建配置 |
| `main/idf_component.yml` | 组件依赖 |
| `partitions.csv` | 分区表 |
| `sdkconfig.defaults` | 默认配置 |
| `verify_model.py` | PC 端验证脚本（需 `.onnx` 模型） |
| `gen_test_data.py` | 生成 PC 预处理 INT8 数据用于交叉验证 |

---

## 🧪 PC 端验证

```bash
# 需要 PPQ 导出的 INT8 ONNX 模型
python verify_model.py
```

预处理公式与 ESP32 驱动完全一致，用来确认模型本身的正确性。

---

## 🔄 迁移指南

要把这个驱动用到自己的二分类模型：

1. **替换模型**: 把 `model.espdl` 替换为你的 ESP-PPQ 导出的模型
2. **调整归一化参数**: 如果你的训练使用不同的 mean/std，修改 `MEAN_255` / `STD_255`
3. **调整图片**: 替换 `t*.jpg` 为你的测试图片
4. **调整输出映射**: 在 `classifier_driver.cpp` 中修改 `label` 和 `class_id` 映射

---

## ⚙️ 依赖

| 组件 | 版本 |
|------|------|
| ESP-IDF | ≥ 5.3（本项目使用 v6.0.1） |
| espressif/esp-dl | latest |
| espressif/esp_new_jpeg | ^1 |
- [ ] 确认 INT8 ONNX 输出与 FP32 ONNX 一致

---

## 使用方式

```bash
# 首次构建
idf.py set-target esp32s3
idf.py build

# 烧录（可能需要按住 BOOT + 按 ENTER）
idf.py -p COMx flash

# 监视串口
idf.py -p COMx monitor

# PC 端验证
python verify_model.py

# 生成 PC 预处理测试数据（重新编译后烧录）
python gen_test_data.py
idf.py build -p COMx flash monitor
```
