#include "classifier_driver.hpp"
#include "dl_model_base.hpp"
#include "dl_image_jpeg.hpp"
#include "dl_image_preprocessor.hpp"
#include "dl_tensor_base.hpp"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include <cmath>
#include <cstring>

static const char *TAG = "classifier";

ClassifierDriver::ClassifierDriver()
    : m_model(nullptr), m_preprocessor(nullptr), m_initialized(false)
{
}

ClassifierDriver::~ClassifierDriver()
{
    if (m_preprocessor) {
        delete static_cast<dl::image::ImagePreprocessor *>(m_preprocessor);
        m_preprocessor = nullptr;
    }
    if (m_model) {
        delete static_cast<dl::Model *>(m_model);
        m_model = nullptr;
    }
}

esp_err_t ClassifierDriver::init()
{
    if (m_initialized) {
        return ESP_OK;
    }

    // 1. Load model from flash partition labeled "model"
    ESP_LOGI(TAG, "Loading model from flash partition...");
    dl::Model *model = new (std::nothrow) dl::Model(
        "model",                                   // partition label
        fbs::MODEL_LOCATION_IN_FLASH_PARTITION,    // location
        0,                                         // max_internal_size (0 = auto)
        dl::MEMORY_MANAGER_GREEDY,                 // memory manager
        nullptr,                                   // encryption key (none)
        true                                       // param_copy (copy to PSRAM for perf)
    );

    if (!model) {
        ESP_LOGE(TAG, "Failed to create model object");
        return ESP_FAIL;
    }

    // Minimize to free metadata after loading
    model->minimize();
    m_model = model;

    // 2. Create image preprocessor
    // Normalization formula: quantized = clamp(((pixel/255 - mean) / std) / scale, -128, 127)
    // where scale = 0.03125 = 1/32 = 2^(-exponent) for exponent=-5
    //
    // ESP-DL ImagePreprocessor expects mean/std in [0,255] range:
    //   mean = [0.485, 0.456, 0.406] × 255 = [123.675, 116.28, 103.53]
    //   std  = [0.229, 0.224, 0.225] × 255 = [58.395, 57.12, 57.375]
    ESP_LOGI(TAG, "Creating image preprocessor...");
    dl::image::ImagePreprocessor *preprocessor =
        new (std::nothrow) dl::image::ImagePreprocessor(
            model,
            {123.675f, 116.28f, 103.53f},   // mean × 255
            {58.395f, 57.12f, 57.375f}      // std × 255
        );

    if (!preprocessor) {
        ESP_LOGE(TAG, "Failed to create image preprocessor");
        delete model;
        m_model = nullptr;
        return ESP_FAIL;
    }

    m_preprocessor = preprocessor;
    m_initialized = true;

    ESP_LOGI(TAG, "Classifier initialized successfully");
    return ESP_OK;
}

classification_result_t ClassifierDriver::infer(const uint8_t *jpg_data, size_t jpg_len)
{
    classification_result_t result = {};
    result.class_id = -1;
    result.score = 0.0f;
    result.probability = 0.0f;
    result.label = "unknown";

    if (!m_initialized || !m_model || !m_preprocessor) {
        ESP_LOGE(TAG, "Classifier not initialized. Call init() first.");
        return result;
    }

    dl::Model *model = static_cast<dl::Model *>(m_model);
    dl::image::ImagePreprocessor *preprocessor =
        static_cast<dl::image::ImagePreprocessor *>(m_preprocessor);

    // 1. Decode JPEG to RGB888
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

    // 2. Preprocess: resize to 224×224 → normalize → quantize into model input
    preprocessor->preprocess(img);

    // 3. Run model inference
    model->run();

    // 4. Get output tensor (shape [2], INT8)
    dl::TensorBase *output = model->get_output();
    if (!output || output->get_size() < 2) {
        ESP_LOGE(TAG, "Invalid model output");
        heap_caps_free(img.data);
        return result;
    }

    int8_t *output_data = output->get_element_ptr<int8_t>();
    int exponent = output->exponent;

    // 5. Dequantize: float_val = int_val × 2^exponent
    float scores[2];
    for (int i = 0; i < 2; i++) {
        scores[i] = dl::dequantize(output_data[i], DL_SCALE(exponent));
    }

    // 6. Softmax with numerical stability
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

    // 7. Argmax
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

    ESP_LOGI(TAG, "Result: class=%s (%d), score=%.4f, prob=%.4f",
             result.label, result.class_id, result.score, result.probability);

    // 8. Free decoded image
    heap_caps_free(img.data);

    return result;
}
