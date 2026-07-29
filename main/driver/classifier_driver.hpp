/**
 * @file classifier_driver.hpp
 * @brief 螺丝/垫圈二分类模型驱动 — 头文件
 *
 * 专为 ESP-DL NHWC INT8 模型设计。
 * 流水线：JPEG 解码 → ImagePreprocessor(resize+crop+量化) → ESP-DL 推理 → softmax
 *
 * 使用方式：
 * @code{.cpp}
 *   ClassifierDriver driver;
 *   driver.init();
 *   classification_result_t res = driver.infer(jpeg_data, jpeg_len);
 *   // class_id == 0 → screw(螺丝),  == 1 → washer(垫圈)
 * @endcode
 */

#pragma once

#include <cstdint>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 单张图片的分类结果
 */
typedef struct {
    int class_id;        /**< 类别: 0=screw(螺丝), 1=washer(垫圈) */
    float score;         /**< 反量化后的原始 logit 值 */
    float probability;   /**< Softmax 概率 (0.0 ~ 1.0) */
    const char *label;   /**< 类别名称 "washer" 或 "screw" */
} classification_result_t;

#ifdef __cplusplus
}
#endif

/**
 * @brief 基于 ESP-DL 的二分类驱动类
 *
 * 职责：
 *   - 从 rodata 加载 model.espdl 模型
 *   - 解码 JPEG 到 RGB888
 *   - resize+crop+归一化+量化 (ImagePreprocessor)
 *   - 执行推理
 *   - 输出反量化 → softmax → argmax → 返回结果
 */
class ClassifierDriver {
public:
    ClassifierDriver();
    ~ClassifierDriver();

    /**
     * @brief 初始化：加载模型 + 创建 ImagePreprocessor
     * @return ESP_OK 成功，ESP_FAIL 失败
     */
    esp_err_t init();

    /**
     * @brief 对 JPEG 图像推理
     * @param jpg_data JPEG 文件数据
     * @param jpg_len  JPEG 数据长度
     */
    classification_result_t infer(const uint8_t *jpg_data, size_t jpg_len);

    /**
     * @brief 直接喂预处理好的 INT8 数据推理（跳过 JPEG 解码和预处理）
     * @param int8_data [224x224x3] INT8 NHWC 数据
     */
    classification_result_t infer_from_preprocessed(const int8_t *int8_data);

private:
    void *m_model;          /*!< dl::Model */
    void *m_preprocessor;   /*!< dl::image::ImagePreprocessor */
    bool  m_initialized;
};
