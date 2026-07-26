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
#include "dl_model_base.hpp"
#include "dl_image_jpeg.hpp"
#include "dl_image_process.hpp"
#include "dl_image_preprocessor.hpp"
#include "dl_tensor_base.hpp"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

static const char *TAG = "classifier";
extern const uint8_t model_espdl[] asm("_binary_model_espdl_start");

static constexpr float MEAN_01[3] = {0.485f, 0.456f, 0.406f};
static constexpr float STD_01[3]  = {0.229f, 0.224f, 0.225f};
static constexpr float MEAN_255[3] = {123.675f, 116.28f, 103.53f};
static constexpr float STD_255[3]  = {58.395f,  57.12f,  57.375f};
static constexpr float SCALE = 32.0f;
static constexpr float INV_255 = 1.0f / 255.0f;

ClassifierDriver::ClassifierDriver()
    : m_model(nullptr), m_preprocessor(nullptr), m_initialized(false), m_is_nchw(false) {}

ClassifierDriver::~ClassifierDriver()
{
    if (m_preprocessor) delete static_cast<dl::image::ImagePreprocessor *>(m_preprocessor);
    if (m_model)        delete static_cast<dl::Model *>(m_model);
}

esp_err_t ClassifierDriver::init()
{
    if (m_initialized) return ESP_OK;

    // 1. 加载模型
    ESP_LOGI(TAG, "Loading model from flash rodata...");
    dl::Model *model = new (std::nothrow) dl::Model(
        (const char *)model_espdl, fbs::MODEL_LOCATION_IN_FLASH_RODATA,
        0, dl::MEMORY_MANAGER_GREEDY, nullptr, true);
    if (!model) { ESP_LOGE(TAG, "Failed to create model"); return ESP_FAIL; }

    if (model->test() == ESP_OK) ESP_LOGI(TAG, "Model self-test PASSED");
    else ESP_LOGW(TAG, "Model self-test skipped (no test data)");

    m_model = model;

    // 2. 检查输入形状，自动适配 NHWC / NCHW
    dl::TensorBase *input = model->get_input();
    if (!input) { ESP_LOGE(TAG, "No input tensor"); delete model; m_model = nullptr; return ESP_FAIL; }

    std::vector<int> s = input->get_shape();
    std::string ss;
    for (size_t i = 0; i < s.size(); i++) { ss += std::to_string(s[i]); if (i+1 < s.size()) ss += "x"; }
    ESP_LOGI(TAG, "Model input: [%s], dtype=%d, exponent=%d", ss.c_str(), (int)input->dtype, (int)input->exponent);

    if (s.size() >= 3 && s[s.size()-1] == 3 && s[s.size()-2] == 224 && s[s.size()-3] == 224) {
        // NHWC: [1, 224, 224, 3] 或 [224, 224, 3] — 使用 ImagePreprocessor
        m_is_nchw = false;
        ESP_LOGI(TAG, "Detected NHWC layout → using ImagePreprocessor");
        m_preprocessor = new dl::image::ImagePreprocessor(model,
            {MEAN_255[0], MEAN_255[1], MEAN_255[2]},
            {STD_255[0],  STD_255[1],  STD_255[2]});
    } else if (s.size() == 3 && s[0] == 3 && s[1] == 224 && s[2] == 224) {
        // NCHW: [3, 224, 224] — 手动量化
        m_is_nchw = true;
        ESP_LOGI(TAG, "Detected NCHW layout → using manual quantization");
    } else {
        ESP_LOGE(TAG, "Unsupported shape. Expected NHWC or NCHW.");
        delete model; m_model = nullptr; return ESP_FAIL;
    }

    m_initialized = true;
    return ESP_OK;
}

void ClassifierDriver::run_diagnostic()
{
    if (!m_initialized || !m_model) { ESP_LOGE(TAG, "Not initialized."); return; }

    dl::Model *model = static_cast<dl::Model *>(m_model);
    const int N = 3 * 224 * 224;

    auto test = [&](const char *label, int8_t val) {
        ESP_LOGI(TAG, "=== Diagnostic: %s ===", label);
        int8_t *b = (int8_t *)heap_caps_malloc(N, MALLOC_CAP_SPIRAM);
        memset(b, val, N);
        dl::TensorBase in({3, 224, 224}, b, -5, dl::DATA_TYPE_INT8, false);
        model->run(&in);
        int8_t *o = model->get_output()->get_element_ptr<int8_t>();
        ESP_LOGI(TAG, "  Output: [%d, %d], exponent=%d", o[0], o[1], (int)model->get_output()->exponent);
        heap_caps_free(b);
    };
    auto test_arr = [&](const char *label, int8_t c0, int8_t c1, int8_t c2) {
        ESP_LOGI(TAG, "=== Diagnostic: %s ===", label);
        int8_t *b = (int8_t *)heap_caps_malloc(N, MALLOC_CAP_SPIRAM);
        for (int i = 0; i < 50176; i++) { b[i]=c0; b[50176+i]=c1; b[100352+i]=c2; }
        dl::TensorBase in({3, 224, 224}, b, -5, dl::DATA_TYPE_INT8, false);
        model->run(&in);
        int8_t *o = model->get_output()->get_element_ptr<int8_t>();
        ESP_LOGI(TAG, "  Output: [%d, %d], exponent=%d", o[0], o[1], (int)model->get_output()->exponent);
        heap_caps_free(b);
    };

    test("all zeros", 0);
    test("all 127", 127);
    test_arr("sim (12,15,16)", 12, 15, 16);
    test_arr("sim (17,21,21)", 17, 21, 21);

    ESP_LOGI(TAG, "=== Diagnostic complete ===");
}

classification_result_t ClassifierDriver::infer(const uint8_t *jpg_data, size_t jpg_len)
{
    classification_result_t r = {}; r.class_id = -1; r.label = "unknown";
    if (!m_initialized || !m_model) { ESP_LOGE(TAG, "Not initialized."); return r; }
    dl::Model *model = static_cast<dl::Model *>(m_model);

    // 1. JPEG 解码
    dl::image::img_t img = dl::image::sw_decode_jpeg(
        {.data = (void *)jpg_data, .data_len = jpg_len}, dl::image::DL_IMAGE_PIX_TYPE_RGB888);
    if (!img.data) { ESP_LOGE(TAG, "JPEG decode failed"); return r; }

    if (m_is_nchw) {
        // ===== NCHW: 手动 resize + 量化 + HWC→CHW =====
        dl::image::img_t res;
        if (img.width != 224 || img.height != 224) {
            float scale_f = 256.0f / (img.width < img.height ? img.width : img.height);
            int nw = (int)(img.width * scale_f), nh = (int)(img.height * scale_f);
            dl::image::img_t tmp = {heap_caps_malloc(nw*nh*3, MALLOC_CAP_DEFAULT), (uint16_t)nw, (uint16_t)nh, img.pix_type};
            dl::image::ImageTransformer().set_src_img(img).set_dst_img(tmp).transform(); heap_caps_free(img.data);
            int cx = (nw-224)/2, cy = (nh-224)/2;
            res = {heap_caps_malloc(224*224*3, MALLOC_CAP_DEFAULT), 224, 224, img.pix_type};
            dl::image::ImageTransformer().set_src_img(tmp).set_dst_img(res).set_src_img_crop_area({cx,cy,cx+224,cy+224}).transform(); heap_caps_free(tmp.data);
        } else res = img;

        int8_t *input = (int8_t *)heap_caps_malloc(3*224*224, MALLOC_CAP_SPIRAM);
        uint8_t *src = (uint8_t *)res.data;
        for (int h = 0; h < 224; h++)
            for (int w = 0; w < 224; w++) {
                int hwc = (h*224+w)*3, chw = h*224+w;
                float r = src[hwc+0]*INV_255, g = src[hwc+1]*INV_255, b = src[hwc+2]*INV_255;
                input[0*50176+chw] = (int8_t)fmaxf(-128,fminf(127,roundf(((r-MEAN_01[0])/STD_01[0])*SCALE)));
                input[1*50176+chw] = (int8_t)fmaxf(-128,fminf(127,roundf(((g-MEAN_01[1])/STD_01[1])*SCALE)));
                input[2*50176+chw] = (int8_t)fmaxf(-128,fminf(127,roundf(((b-MEAN_01[2])/STD_01[2])*SCALE)));
            }
        heap_caps_free(res.data);
        { dl::TensorBase in({3,224,224}, input, -5, dl::DATA_TYPE_INT8, false); model->run(&in); }
        heap_caps_free(input);
    } else {
        // ===== NHWC: ImagePreprocessor 自动完成 resize + normalize + quantize =====
        auto *pp = static_cast<dl::image::ImagePreprocessor *>(m_preprocessor);
        pp->preprocess(img); heap_caps_free(img.data);
        model->run();
    }

    // 输出 + 反量化 + softmax + argmax
    dl::TensorBase *out = model->get_output();
    if (!out || out->get_size() < 2) { ESP_LOGE(TAG, "Invalid output"); return r; }
    int8_t *od = out->get_element_ptr<int8_t>(); int e = out->exponent;
    ESP_LOGI(TAG, "Raw output: [%d,%d] exp=%d", od[0], od[1], e);
    float s[2]; for (int i=0;i<2;i++) s[i]=dl::dequantize(od[i], DL_SCALE(e));
    float mx = (s[0]>s[1])?s[0]:s[1], e0=expf(s[0]-mx), e1=expf(s[1]-mx);
    float p0=e0/(e0+e1), p1=e1/(e0+e1);
    if (p0>=p1) { r.class_id=0; r.score=s[0]; r.probability=p0; r.label="washer"; }
    else        { r.class_id=1; r.score=s[1]; r.probability=p1; r.label="screw";  }
    ESP_LOGI(TAG, "Result: %s(%d) score=%.4f prob=%.2f%%",
             r.label, r.class_id, r.score, r.probability*100.0f);
    return r;
}
