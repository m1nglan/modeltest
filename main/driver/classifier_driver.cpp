/**
 * @file classifier_driver.cpp
 * @brief 螺丝/垫圈二分类模型驱动 — 实现
 *
 * 专为 NHWC [1x224x224x3] INT8 模型设计。
 * 流水线：JPEG 解码 → ImagePreprocessor(resize+crop+量化) → ESP-DL 推理 → softmax
 *
 * 预处理公式：
 *   quantized = clamp(round(((pixel/255 - mean) / std) / scale), -128, 127)
 *   其中 mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225], scale=2^(-6)=0.015625
 *
 * ESP-DL 量化方式：对称量化 (zero_point=0)
 *   quantized = round(float_value / (2^exponent))
 *   dequantized = quantized * (2^exponent)
 */

#include "classifier_driver.hpp"
#include "dl_model_base.hpp"
#include "dl_image_jpeg.hpp"
#include "dl_image_preprocessor.hpp"
#include "dl_tensor_base.hpp"
#include "esp_log.h"
#include <cmath>

static const char *TAG = "classifier";
extern const uint8_t model_espdl[] asm("_binary_model_espdl_start");

/* ImageNet 归一化参数（[0, 255] 范围，ImagePreprocessor 要求） */
static constexpr float MEAN_255[3] = {123.675f, 116.28f, 103.53f};
static constexpr float STD_255[3]  = {58.395f,  57.12f,  57.375f};

ClassifierDriver::ClassifierDriver()
    : m_model(nullptr), m_preprocessor(nullptr), m_initialized(false) {}

ClassifierDriver::~ClassifierDriver()
{
    if (m_preprocessor) delete static_cast<dl::image::ImagePreprocessor *>(m_preprocessor);
    if (m_model)        delete static_cast<dl::Model *>(m_model);
}

esp_err_t ClassifierDriver::init()
{
    if (m_initialized) return ESP_OK;

    // 从 rodata 加载模型
    dl::Model *model = new (std::nothrow) dl::Model(
        (const char *)model_espdl, fbs::MODEL_LOCATION_IN_FLASH_RODATA,
        0, dl::MEMORY_MANAGER_GREEDY, nullptr, true);
    if (!model) { ESP_LOGE(TAG, "Failed to create model"); return ESP_FAIL; }
    m_model = model;

    // 创建 ImagePreprocessor（自动从模型读取 input shape 和 exponent）
    dl::TensorBase *input = model->get_input();
    if (!input) { ESP_LOGE(TAG, "No input tensor"); delete model; m_model = nullptr; return ESP_FAIL; }

    m_preprocessor = new dl::image::ImagePreprocessor(model,
        {MEAN_255[0], MEAN_255[1], MEAN_255[2]},
        {STD_255[0],  STD_255[1],  STD_255[2]});

    m_initialized = true;
    ESP_LOGI(TAG, "Classifier initialized");
    return ESP_OK;
}

classification_result_t ClassifierDriver::infer(const uint8_t *jpg_data, size_t jpg_len)
{
    classification_result_t r = {}; r.class_id = -1; r.label = "unknown";
    if (!m_initialized || !m_model) return r;
    dl::Model *model = static_cast<dl::Model *>(m_model);

    // 1. JPEG 解码
    dl::image::img_t img = dl::image::sw_decode_jpeg(
        {.data = (void *)jpg_data, .data_len = jpg_len}, dl::image::DL_IMAGE_PIX_TYPE_RGB888);
    if (!img.data) { ESP_LOGE(TAG, "JPEG decode failed"); return r; }

    // 2. 预处理：resize + center crop + 归一化 + 量化（写入模型输入 tensor）
    auto *pp = static_cast<dl::image::ImagePreprocessor *>(m_preprocessor);
    pp->preprocess(img);
    heap_caps_free(img.data);

    // 3. 推理
    model->run();

    // 4. 输出反量化 + softmax + argmax
    dl::TensorBase *out = model->get_output();
    if (!out || out->get_size() < 2) return r;
    int8_t *od = out->get_element_ptr<int8_t>();
    int e = out->exponent;
    float s[2]; for (int i=0;i<2;i++) s[i]=dl::dequantize(od[i], DL_SCALE(e));
    float mx = (s[0]>s[1])?s[0]:s[1], e0=expf(s[0]-mx), e1=expf(s[1]-mx);
    float p0=e0/(e0+e1), p1=e1/(e0+e1);
    if (p0>=p1) { r.class_id=0; r.score=s[0]; r.probability=p0; r.label="washer"; }
    else        { r.class_id=1; r.score=s[1]; r.probability=p1; r.label="screw";  }
    ESP_LOGI(TAG, "Result: %s(%d) score=%.4f prob=%.2f%%",
             r.label, r.class_id, r.score, r.probability*100.0f);
    return r;
}

classification_result_t ClassifierDriver::infer_from_preprocessed(const int8_t *int8_data)
{
    classification_result_t r = {}; r.class_id = -1; r.label = "unknown";
    if (!m_initialized || !m_model) return r;
    dl::Model *model = static_cast<dl::Model *>(m_model);

    // 直接写入模型输入 tensor（跳过 JPEG 解码和预处理，用于调试对比）
    dl::TensorBase *in_t = model->get_input();
    memcpy(in_t->get_element_ptr<int8_t>(), int8_data, in_t->get_size() * sizeof(int8_t));

    model->run();

    // 输出处理（与 infer() 相同）
    dl::TensorBase *out = model->get_output();
    if (!out || out->get_size() < 2) return r;
    int8_t *od = out->get_element_ptr<int8_t>();
    int e = out->exponent;
    float s[2]; for (int i=0;i<2;i++) s[i]=dl::dequantize(od[i], DL_SCALE(e));
    float mx = (s[0]>s[1])?s[0]:s[1], e0=expf(s[0]-mx), e1=expf(s[1]-mx);
    float p0=e0/(e0+e1), p1=e1/(e0+e1);
    if (p0>=p1) { r.class_id=0; r.score=s[0]; r.probability=p0; r.label="washer"; }
    else        { r.class_id=1; r.score=s[1]; r.probability=p1; r.label="screw";  }
    ESP_LOGI(TAG, "Result: %s(%d) score=%.4f prob=%.2f%%",
             r.label, r.class_id, r.score, r.probability*100.0f);
    return r;
}
