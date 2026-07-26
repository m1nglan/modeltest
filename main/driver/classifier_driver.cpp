#include "classifier_driver.hpp"
#include "dl_model_base.hpp"
#include "dl_image_jpeg.hpp"
#include "dl_image_process.hpp"
#include "dl_tensor_base.hpp"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

static const char *TAG = "classifier";

/* Normalization parameters (same as training) */
static constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
static constexpr float STD[3]  = {0.229f, 0.224f, 0.225f};
/* Quantization scale = 2^(-exponent) = 2^5 = 32 for exponent=-5 */

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

    // Run built-in model self-test (if exported with test data in PPQ)
    esp_err_t test_ret = model->test();
    if (test_ret == ESP_OK) {
        ESP_LOGI(TAG, "Model self-test PASSED");
    } else {
        ESP_LOGW(TAG, "Model self-test skipped or failed (model may not have embedded test data)");
    }

    // NOTE: minimize() frees metadata but may also release model weights,
    // causing all outputs to be identical (just bias). Disabled for correctness.
    // model->minimize();
    m_model = model;

    // 2. Inspect input tensor shape (expected: [3, 224, 224] NCHW)
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

    if (in_shape.size() != 3 || in_shape[0] != 3 || in_shape[1] != 224 || in_shape[2] != 224) {
        ESP_LOGE(TAG, "Unexpected input shape. Expected [3, 224, 224] (NCHW)");
        delete model;
        m_model = nullptr;
        return ESP_FAIL;
    }

    m_initialized = true;

    ESP_LOGI(TAG, "Classifier initialized successfully. Input NCHW=[3,%d,%d]",
             in_shape[1], in_shape[2]);
    return ESP_OK;
}

classification_result_t ClassifierDriver::infer(const uint8_t *jpg_data, size_t jpg_len)
{
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

    // ====== 1. Decode JPEG to RGB888 ======
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

    // ====== 2. Resize to 224×224 if needed ======
    dl::image::img_t resized;
    if (img.width != 224 || img.height != 224) {
        // Allocate destination buffer
        resized.data = heap_caps_malloc(224 * 224 * 3, MALLOC_CAP_DEFAULT);
        resized.width = 224;
        resized.height = 224;
        resized.pix_type = img.pix_type;

        dl::image::ImageTransformer transformer;
        transformer.set_src_img(img)
                  .set_dst_img(resized)
                  .transform();

        // Free original, use resized
        heap_caps_free(img.data);
        ESP_LOGI(TAG, "Resized to 224x224");
    } else {
        resized = img;  // already 224x224, use directly
        ESP_LOGI(TAG, "Image already 224x224, no resize needed");
    }

    // ====== 3. Manual NCHW quantization into a temp buffer ======
    // Formula: quantized = clamp(round(((pixel/255 - mean) / std) / scale), -128, 127)
    // scale = 2^(-exponent) = 2^5 = 32
    // Combined: q = clamp(round(((pixel/255 - mean) / std) * 32), -128, 127)
    constexpr float INV_255 = 1.0f / 255.0f;
    constexpr float SCALE = 32.0f;  // 1 / 0.03125

    // Allocate a separate buffer for preprocessed INT8 data (NCHW layout)
    int8_t *input_data = (int8_t *)heap_caps_malloc(3 * 224 * 224, MALLOC_CAP_SPIRAM);
    if (!input_data) {
        ESP_LOGE(TAG, "Failed to allocate input buffer");
        heap_caps_free(resized.data);
        return result;
    }

    uint8_t *src = static_cast<uint8_t *>(resized.data);

    // DEBUG: print first 4 pixels of source
    ESP_LOGI(TAG, "Source pixels[0..3]: R=%d %d %d %d, G=%d %d %d %d, B=%d %d %d %d",
        src[0], src[3], src[6], src[9],
        src[1], src[4], src[7], src[10],
        src[2], src[5], src[8], src[11]);

    for (int h = 0; h < 224; h++) {
        for (int w = 0; w < 224; w++) {
            int hwc_idx = (h * 224 + w) * 3;
            float r = src[hwc_idx + 0] * INV_255;
            float g = src[hwc_idx + 1] * INV_255;
            float b = src[hwc_idx + 2] * INV_255;

            // Normalize → quantize → clamp → NCHW layout
            int chw_base = h * 224 + w;
            int q_r = (int)roundf(((r - MEAN[0]) / STD[0]) * SCALE);
            int q_g = (int)roundf(((g - MEAN[1]) / STD[1]) * SCALE);
            int q_b = (int)roundf(((b - MEAN[2]) / STD[2]) * SCALE);

            input_data[0 * 50176 + chw_base] = (int8_t)fmaxf(-128.0f, fminf(127.0f, (float)q_r));
            input_data[1 * 50176 + chw_base] = (int8_t)fmaxf(-128.0f, fminf(127.0f, (float)q_g));
            input_data[2 * 50176 + chw_base] = (int8_t)fmaxf(-128.0f, fminf(127.0f, (float)q_b));
        }
    }

    // DEBUG: print first 4 quantized values per channel
    ESP_LOGI(TAG, "Quantized ch0[0..3]: %d %d %d %d",
        input_data[0], input_data[1], input_data[2], input_data[3]);
    ESP_LOGI(TAG, "Quantized ch1[0..3]: %d %d %d %d",
        input_data[50176], input_data[50177], input_data[50178], input_data[50179]);

    heap_caps_free(resized.data);

    // ====== 4. Create input tensor and run model inference ======
    // Use the run(TensorBase *input) API to explicitly pass the input
    {
        // Wrap our buffer in a TensorBase (deep=false → no copy)
        dl::TensorBase input_tensor(
            {3, 224, 224},           // shape (NCHW)
            input_data,              // preprocessed INT8 data
            -5,                      // exponent (matches model input)
            dl::DATA_TYPE_INT8,      // dtype
            false                    // deep = false, use pointer directly
        );
        model->run(&input_tensor);
    }

    // Free our input buffer after inference
    heap_caps_free(input_data);

    // ====== 5. Get output tensor (shape [2], INT8) ======
    dl::TensorBase *output = model->get_output();
    if (!output || output->get_size() < 2) {
        ESP_LOGE(TAG, "Invalid model output");
        return result;
    }

    int8_t *output_data = output->get_element_ptr<int8_t>();
    int exponent = output->exponent;

    ESP_LOGI(TAG, "Raw output INT8: [%d, %d], exponent=%d",
             output_data[0], output_data[1], exponent);

    // ====== 6. Dequantize: float_val = int_val × 2^exponent ======
    float scores[2];
    for (int i = 0; i < 2; i++) {
        scores[i] = dl::dequantize(output_data[i], DL_SCALE(exponent));
    }

    // ====== 7. Softmax with numerical stability ======
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

    // ====== 8. Argmax ======
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
