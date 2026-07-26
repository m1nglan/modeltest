/**
 * @file classifier_driver.cpp
 * @brief 螺丝/垫圈二分类模型驱动 — 实现
 *
 * 预处理公式（从训练代码继承）：
 *   quantized = clamp(round(((pixel/255 - mean) / std) / scale), -128, 127)
 *   其中 mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225], scale=2^(-5)=0.03125
 *
 * ESP-DL 量化方式：对称量化 (zero_point=0)
 *   quantized = round(float_value / (2^exponent))
 *   dequantized = quantized * (2^exponent)
 */

#include "classifier_driver.hpp"
#include "dl_model_base.hpp"    // dl::Model, Model 加载和推理 API
#include "dl_image_jpeg.hpp"    // sw_decode_jpeg: 软件 JPEG 解码
#include "dl_image_process.hpp" // ImageTransformer: resize
#include "dl_tensor_base.hpp"   // TensorBase: ESP-DL 张量操作
#include "esp_log.h"            // ESP_LOGI, ESP_LOGE 日志
#include "esp_heap_caps.h"      // heap_caps_malloc / free
#include <cmath>                // roundf, expf, fmaxf, fminf
#include <cstring>              // memset
#include <string>               // std::string
#include <vector>               // std::vector

/* 日志标签 */
static const char *TAG = "classifier";

/*
 * 模型二进制文件 model.espdl（约 2.4MB）通过 CMakeLists.txt 中的
 * target_add_aligned_binary_data 嵌入到固件的 .rodata 段。
 * 链接器符号：_binary_model_espdl_start
 */
extern const uint8_t model_espdl[] asm("_binary_model_espdl_start");

/*
 * 图像归一化参数（与训练时完全一致）
 * 训练时 PyTorch 代码：
 *   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
 *
 * ESP-DL 要求 mean/std 在 [0,255] 范围，但这里我们手动实现预处理，
 * 所以直接用训练时的 [0,1] 范围值。
 */
static constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
static constexpr float STD[3]  = {0.229f, 0.224f, 0.225f};

/*
 * 量化缩放因子
 * 模型输入 exponent=-5 → scale = 2^(-(-5)) = 32
 * 对应公式中的 /scale: 相当于乘以 1/0.03125 = 32
 */
static constexpr float SCALE = 32.0f;
static constexpr float INV_255 = 1.0f / 255.0f;  // pixel/255 预计算

/* ==================== 构造函数 / 析构函数 ==================== */

ClassifierDriver::ClassifierDriver()
    : m_model(nullptr), m_initialized(false)
{
}

ClassifierDriver::~ClassifierDriver()
{
    if (m_model) {
        delete static_cast<dl::Model *>(m_model);
        m_model = nullptr;
    }
}

/* ==================== 初始化 ==================== */

esp_err_t ClassifierDriver::init()
{
    if (m_initialized) {
        return ESP_OK;  // 防止重复初始化
    }

    // ====== 1. 加载模型 ======
    ESP_LOGI(TAG, "Loading model from flash rodata...");
    dl::Model *model = new (std::nothrow) dl::Model(
        (const char *)model_espdl,           // rodata 中模型数据的起始地址
        fbs::MODEL_LOCATION_IN_FLASH_RODATA, // 模型位置：嵌入在 app 的 rodata 段
        0,                                   // max_internal_size: 0=自动
        dl::MEMORY_MANAGER_GREEDY,           // 内存管理器：贪婪模式（优先用 PSRAM）
        nullptr,                             // 加密密钥：无
        true                                 // param_copy: 从 flash 拷贝到 PSRAM 提高性能
    );

    if (!model) {
        ESP_LOGE(TAG, "Failed to create model object");
        return ESP_FAIL;
    }

    /*
     * 运行模型内置自检（仅在 PPQ 导出时启用 export_test_values 才有效）
     * 测试失败不阻塞初始化，只是提示没有嵌入测试数据。
     */
    esp_err_t test_ret = model->test();
    if (test_ret == ESP_OK) {
        ESP_LOGI(TAG, "Model self-test PASSED");
    } else {
        ESP_LOGW(TAG, "Model self-test skipped or failed (no embedded test data)");
    }

    /*
     * 注意：minimize() 会释放 name-to-index 映射表以节省内存，
     * 但可能导致权重被释放，使所有输出变为相同的 bias-only 值。
     * 这里禁用以确保推理正确性。
     */
    // model->minimize();
    m_model = model;

    // ====== 2. 检查输入张量形状 ======
    dl::TensorBase *input = model->get_input();
    if (!input) {
        ESP_LOGE(TAG, "Model has no input tensor");
        delete model;
        m_model = nullptr;
        return ESP_FAIL;
    }

    std::vector<int> in_shape = input->get_shape();
    std::string shape_str;
    for (size_t i = 0; i < in_shape.size(); i++) {
        shape_str += std::to_string(in_shape[i]);
        if (i < in_shape.size() - 1) shape_str += "x";
    }
    ESP_LOGI(TAG, "Model input shape: [%s], dtype=%d, exponent=%d",
             shape_str.c_str(), (int)input->dtype, (int)input->exponent);

    /*
     * 验证输入形状是否为 [3, 224, 224] NCHW 格式
     * ESP-DL 通常使用 NHWC [1, H, W, 3]，但这个模型是 NCHW [3, H, W]
     */
    if (in_shape.size() != 3 || in_shape[0] != 3 || in_shape[1] != 224 || in_shape[2] != 224) {
        ESP_LOGE(TAG, "Unexpected input shape. Expected [3, 224, 224] (NCHW)");
        delete model;
        m_model = nullptr;
        return ESP_FAIL;
    }

    m_initialized = true;
    ESP_LOGI(TAG, "Classifier initialized. Input NCHW=[3,%d,%d]",
             in_shape[1], in_shape[2]);
    return ESP_OK;
}

/* ==================== 诊断方法 ==================== */

void ClassifierDriver::run_diagnostic()
{
    if (!m_initialized || !m_model) {
        ESP_LOGE(TAG, "Classifier not initialized.");
        return;
    }

    dl::Model *model = static_cast<dl::Model *>(m_model);
    const int input_size = 3 * 224 * 224;  // 输入张量的元素总数

    // ---- 测试 1: 全 0 输入 ----
    ESP_LOGI(TAG, "=== Diagnostic: input = all zeros ===");
    int8_t *zeros = (int8_t *)heap_caps_malloc(input_size, MALLOC_CAP_SPIRAM);
    memset(zeros, 0, input_size);
    {
        dl::TensorBase input_tensor({3, 224, 224}, zeros, -5, dl::DATA_TYPE_INT8, false);
        model->run(&input_tensor);
        dl::TensorBase *out = model->get_output();
        int8_t *odata = out->get_element_ptr<int8_t>();
        ESP_LOGI(TAG, "  Output: [%d, %d], exponent=%d", odata[0], odata[1], (int)out->exponent);
    }

    // ---- 测试 2: 全 127 输入（最大正 INT8 值） ----
    ESP_LOGI(TAG, "=== Diagnostic: input = all 127 ===");
    int8_t *ones = (int8_t *)heap_caps_malloc(input_size, MALLOC_CAP_SPIRAM);
    memset(ones, 127, input_size);
    {
        dl::TensorBase input_tensor({3, 224, 224}, ones, -5, dl::DATA_TYPE_INT8, false);
        model->run(&input_tensor);
        dl::TensorBase *out = model->get_output();
        int8_t *odata = out->get_element_ptr<int8_t>();
        ESP_LOGI(TAG, "  Output: [%d, %d], exponent=%d", odata[0], odata[1], (int)out->exponent);
    }

    // ---- 测试 3: 渐变输入（从 0 线性增加到 127） ----
    ESP_LOGI(TAG, "=== Diagnostic: input = gradient ===");
    int8_t *grad = (int8_t *)heap_caps_malloc(input_size, MALLOC_CAP_SPIRAM);
    for (int i = 0; i < input_size; i++) {
        grad[i] = (int8_t)((i * 127) / input_size);
    }
    {
        dl::TensorBase input_tensor({3, 224, 224}, grad, -5, dl::DATA_TYPE_INT8, false);
        model->run(&input_tensor);
        dl::TensorBase *out = model->get_output();
        int8_t *odata = out->get_element_ptr<int8_t>();
        ESP_LOGI(TAG, "  Output: [%d, %d], exponent=%d", odata[0], odata[1], (int)out->exponent);
    }

    heap_caps_free(zeros);
    heap_caps_free(ones);
    heap_caps_free(grad);

    ESP_LOGI(TAG, "=== Diagnostic complete ===");
}

void ClassifierDriver::run_diagnostic()
{
    if (!m_initialized || !m_model) {
        ESP_LOGE(TAG, "Classifier not initialized.");
        return;
    }

    dl::Model *model = static_cast<dl::Model *>(m_model);
    const int input_size = 3 * 224 * 224;

    // Test 1: All zeros
    ESP_LOGI(TAG, "=== Diagnostic: input = all zeros ===");
    int8_t *zeros = (int8_t *)heap_caps_malloc(input_size, MALLOC_CAP_SPIRAM);
    memset(zeros, 0, input_size);
    {
        dl::TensorBase input_tensor({3, 224, 224}, zeros, -5, dl::DATA_TYPE_INT8, false);
        model->run(&input_tensor);
        dl::TensorBase *out = model->get_output();
        int8_t *odata = out->get_element_ptr<int8_t>();
        ESP_LOGI(TAG, "  Output: [%d, %d], exponent=%d", odata[0], odata[1], (int)out->exponent);
    }

    // Test 2: All 127 (max positive INT8)
    ESP_LOGI(TAG, "=== Diagnostic: input = all 127 ===");
    int8_t *ones = (int8_t *)heap_caps_malloc(input_size, MALLOC_CAP_SPIRAM);
    memset(ones, 127, input_size);
    {
        dl::TensorBase input_tensor({3, 224, 224}, ones, -5, dl::DATA_TYPE_INT8, false);
        model->run(&input_tensor);
        dl::TensorBase *out = model->get_output();
        int8_t *odata = out->get_element_ptr<int8_t>();
        ESP_LOGI(TAG, "  Output: [%d, %d], exponent=%d", odata[0], odata[1], (int)out->exponent);
    }

    // Test 3: Gradient (increasing row by row)
    ESP_LOGI(TAG, "=== Diagnostic: input = gradient ===");
    int8_t *grad = (int8_t *)heap_caps_malloc(input_size, MALLOC_CAP_SPIRAM);
    for (int i = 0; i < input_size; i++) {
        grad[i] = (int8_t)((i * 127) / input_size);
    }
    {
        dl::TensorBase input_tensor({3, 224, 224}, grad, -5, dl::DATA_TYPE_INT8, false);
        model->run(&input_tensor);
        dl::TensorBase *out = model->get_output();
        int8_t *odata = out->get_element_ptr<int8_t>();
        ESP_LOGI(TAG, "  Output: [%d, %d], exponent=%d", odata[0], odata[1], (int)out->exponent);
    }

    heap_caps_free(zeros);
    heap_caps_free(ones);
    heap_caps_free(grad);

    ESP_LOGI(TAG, "=== Diagnostic complete ===");
}

/* ==================== 推理主方法 ==================== */

classification_result_t ClassifierDriver::infer(const uint8_t *jpg_data, size_t jpg_len)
{
    /* 初始化结果结构体（默认值：unknown） */
    classification_result_t result = {};
    result.class_id = -1;
    result.score = 0.0f;
    result.probability = 0.0f;
    result.label = "unknown";

    if (!m_initialized || !m_model) {
        ESP_LOGE(TAG, "Classifier not initialized. Call init() first.");
        return result;
    }

    dl::Model *model = static_cast<dl::Model *>(m_model);

    /*
     * ====== 步骤 1: JPEG 解码 ======
     * 输入：嵌入在 rodata 中的 JPEG 文件字节流
     * 输出：dl::image::img_t 结构体，包含 RGB888 格式的图像数据
     *       - data:   uint8_t 像素数据，HWC 布局
     *       - width:  图像宽度
     *       - height: 图像高度
     *       - pix_type: 像素格式 (DL_IMAGE_PIX_TYPE_RGB888)
     *
     * sw_decode_jpeg 使用软件 JPEG 解码器（库：esp_new_jpeg）
     */
    dl::image::jpeg_img_t jpeg = {
        .data = const_cast<void *>(static_cast<const void *>(jpg_data)),
        .data_len = jpg_len
    };
    dl::image::img_t img = dl::image::sw_decode_jpeg(
        jpeg, dl::image::DL_IMAGE_PIX_TYPE_RGB888
    );

    if (!img.data) {
        ESP_LOGE(TAG, "Failed to decode JPEG image");
        return result;
    }

    ESP_LOGI(TAG, "Decoded JPEG: %dx%d, type=%d", img.width, img.height, img.pix_type);

    /*
     * ====== 步骤 2: Resize 到 224×224 ======
     * 模型输入固定为 224×224，如果原始图片不是这个尺寸，
     * 使用 ImageTransformer 做 resize。
     *
     * 注意：这里做的是拉伸 resize（不保持宽高比），
     * 因为训练时使用的是 transforms.Resize(256) + CenterCrop(224)。
     * 更准确的做法是先 resize 到 256 再 center crop 到 224，
     * 但对于测试用途简单 resize 也够用。
     */
    dl::image::img_t resized;
    if (img.width != 224 || img.height != 224) {
        resized.data = heap_caps_malloc(224 * 224 * 3, MALLOC_CAP_DEFAULT);
        resized.width = 224;
        resized.height = 224;
        resized.pix_type = img.pix_type;

        dl::image::ImageTransformer transformer;
        transformer.set_src_img(img)
                  .set_dst_img(resized)
                  .transform();

        heap_caps_free(img.data);  // 释放原图
        ESP_LOGI(TAG, "Resized to 224x224");
    } else {
        resized = img;  // 已经是 224x224，直接使用
        ESP_LOGI(TAG, "Image already 224x224, no resize needed");
    }

    /*
     * ====== 步骤 3: 手动量化 + NCHW 布局转换 ======
     *
     * 预处理公式：
     *   quantized = clamp(round(((pixel/255 - mean) / std) / scale), -128, 127)
     *
     * 其中：
     *   - pixel/255： 将 [0,255] 像素值归一化到 [0,1]
     *   - (x-mean)/std： ImageNet 标准归一化
     *   - / scale：     除以量化 scale (= 0.03125)，等价于 ×32
     *   - clamp：       限制到 INT8 范围 [-128, 127]
     *
     * 布局转换：
     *   原图是 HWC 布局 [H=224, W=224, C=3]
     *   模型需要 NCHW 布局 [C=3, H=224, W=224]
     *   所以 output[0][h][w] = R, output[1][h][w] = G, output[2][h][w] = B
     *
     * 内存大小：3 × 224 × 224 = 150528 字节
     * 索引常量：50176 = 224 × 224（单通道元素数）
     */

    /* 分配独立的量化缓冲区（放在 PSRAM 中） */
    int8_t *input_data = (int8_t *)heap_caps_malloc(3 * 224 * 224, MALLOC_CAP_SPIRAM);
    if (!input_data) {
        ESP_LOGE(TAG, "Failed to allocate input buffer");
        heap_caps_free(resized.data);
        return result;
    }

    uint8_t *src = static_cast<uint8_t *>(resized.data);

    /* 调试：打印前 4 个像素的 RGB 值 */
    ESP_LOGI(TAG, "Source pixels[0..3]: R=%d %d %d %d, G=%d %d %d %d, B=%d %d %d %d",
        src[0], src[3], src[6], src[9],
        src[1], src[4], src[7], src[10],
        src[2], src[5], src[8], src[11]);

    /* 遍历每个像素，量化并写入 NCHW 布局 */
    for (int h = 0; h < 224; h++) {
        for (int w = 0; w < 224; w++) {
            int hwc_idx = (h * 224 + w) * 3;  // HWC 布局中的像素偏移
            float r = src[hwc_idx + 0] * INV_255;  // R 通道
            float g = src[hwc_idx + 1] * INV_255;  // G 通道
            float b = src[hwc_idx + 2] * INV_255;  // B 通道

            /* 归一化 + 量化 + clamp */
            int chw_base = h * 224 + w;            // NCHW 布局中的空间位置
            int q_r = (int)roundf(((r - MEAN[0]) / STD[0]) * SCALE);
            int q_g = (int)roundf(((g - MEAN[1]) / STD[1]) * SCALE);
            int q_b = (int)roundf(((b - MEAN[2]) / STD[2]) * SCALE);

            /*
             * NCHW 布局：通道 0=R, 通道 1=G, 通道 2=B
             * 每个通道有 50176 = 224×224 个元素
             */
            input_data[0 * 50176 + chw_base] = (int8_t)fmaxf(-128.0f, fminf(127.0f, (float)q_r));
            input_data[1 * 50176 + chw_base] = (int8_t)fmaxf(-128.0f, fminf(127.0f, (float)q_g));
            input_data[2 * 50176 + chw_base] = (int8_t)fmaxf(-128.0f, fminf(127.0f, (float)q_b));
        }
    }

    /* 调试：打印前 4 个量化后的值 */
    ESP_LOGI(TAG, "Quantized ch0[0..3]: %d %d %d %d",
        input_data[0], input_data[1], input_data[2], input_data[3]);
    ESP_LOGI(TAG, "Quantized ch1[0..3]: %d %d %d %d",
        input_data[50176], input_data[50177], input_data[50178], input_data[50179]);

    heap_caps_free(resized.data);  // resize 后的图像不再需要

    /*
     * ====== 步骤 4: 执行推理 ======
     * 创建 TensorBase 包装我们的量化数据，
     * 通过 model->run(TensorBase*) API 显式传入输入。
     * deep=false 表示 TensorBase 不拷贝数据，直接使用我们的缓冲区。
     */
    {
        dl::TensorBase input_tensor(
            {3, 224, 224},           // shape: NCHW
            input_data,              // 量化后的 INT8 数据
            -5,                      // exponent（与模型输入一致）
            dl::DATA_TYPE_INT8,      // 数据类型
            false                    // deep=false: 不拷贝，直接使用指针
        );
        model->run(&input_tensor);   // 执行完整模型推理
    }

    heap_caps_free(input_data);  // 推理完成后释放输入缓冲区

    /*
     * ====== 步骤 5: 获取输出 ======
     * 模型输出是形状 [2] 的 INT8 张量，
     * 对应两个类别的 logit 值。
     * - output[0] = screw 的 logit
     * - output[1] = washer 的 logit
     */
    dl::TensorBase *output = model->get_output();
    if (!output || output->get_size() < 2) {
        ESP_LOGE(TAG, "Invalid model output");
        return result;
    }

    int8_t *output_data = output->get_element_ptr<int8_t>();
    int exponent = output->exponent;

    ESP_LOGI(TAG, "Raw output INT8: [%d, %d], exponent=%d",
             output_data[0], output_data[1], exponent);

    /*
     * ====== 步骤 6: 反量化 ======
     * dequantized = quantized × 2^exponent
     * DL_SCALE(e) = 2^e
     */
    float scores[2];
    for (int i = 0; i < 2; i++) {
        scores[i] = dl::dequantize(output_data[i], DL_SCALE(exponent));
    }

    /*
     * ====== 步骤 7: Softmax ======
     * 使用数值稳定版本的 softmax：
     *   p_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
     * 减去最大值防止 exp 溢出。
     */
    float max_score = (scores[0] > scores[1]) ? scores[0] : scores[1];
    float exp_sum = 0.0f;
    float exp_vals[2];
    for (int i = 0; i < 2; i++) {
        exp_vals[i] = expf(scores[i] - max_score);
        exp_sum += exp_vals[i];
    }
    float probs[2];
    for (int i = 0; i < 2; i++) {
        probs[i] = exp_vals[i] / exp_sum;
    }

    /*
     * ====== 步骤 8: 取 argmax ======
     * class_id=0 → screw（螺丝）
     * class_id=1 → washer（垫圈）
     */
    if (probs[0] >= probs[1]) {
        result.class_id = 0;
        result.score = scores[0];
        result.probability = probs[0];
        result.label = "screw";
    } else {
        result.class_id = 1;
        result.score = scores[1];
        result.probability = probs[1];
        result.label = "washer";
    }

    ESP_LOGI(TAG, "Result: class=%s (%d), score=%.4f, prob=%.4f%%",
             result.label, result.class_id, result.score, result.probability * 100.0f);

    return result;
}
