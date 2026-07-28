/**
 * @file classifier_driver.hpp
 * @brief 螺丝/垫圈二分类模型驱动 — 头文件
 *
 * 封装 ESP-DL 模型加载、JPEG 解码、图像预处理、推理执行、
 * 输出反量化和后处理的完整流水线。
 *
 * 使用方式：
 * @code{.cpp}
 *   ClassifierDriver driver;
 *   driver.init();
 *   classification_result_t res = driver.infer(jpeg_data, jpeg_len);
 *   // res.class_id == 0 → "screw",  == 1 → "washer"
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
    int class_id;        /**< 类别 ID: 0=screw(螺丝), 1=washer(垫圈) */
    float score;         /**< 反量化后的原始 logit 值 */
    float probability;   /**< Softmax 概率 (0.0 ~ 1.0) */
    const char *label;   /**< 类别名称: "screw" 或 "washer" */
} classification_result_t;

#ifdef __cplusplus
}
#endif

/**
 * @brief 基于 ESP-DL 的二分类驱动类
 *
 * 职责：
 *   - 从 rodata 加载 model.espdl 模型
 *   - 解码 JPEG 图像到 RGB888
 *   - resize 到 224×224
 *   - 应用 ImageNet 归一化 + 量化 (pixel/255 → (x-mean)/std → /scale → clamp)
 *   - 转换 HWC → NCHW 布局（模型要求）
 *   - 执行 ESP-DL 推理
 *   - 反量化输出 → softmax → argmax → 返回结果
 */
class ClassifierDriver {
public:
    /**
     * @brief 构造函数（不执行任何初始化）
     * @note init() 必须在使用 infer() 之前调用
     */
    ClassifierDriver();

    /**
     * @brief 析构函数（释放模型和所有分配的资源）
     */
    ~ClassifierDriver();

    /**
     * @brief 初始化分类器
     *        - 从 rodata 加载 model.espdl
     *        - 读取并验证输入张量形状 [3,224,224]
     *        - 运行内置测试（如果模型包含测试数据）
     * @return ESP_OK  成功
     * @return ESP_FAIL 失败（模型加载或形状不匹配）
     */
    esp_err_t init();

    /**
     * @brief 对一幅 JPEG 图像执行推理
     *
     * 完整流水线：
     *   JPEG 解码 → resize → 归一化+量化 → HWC→NCHW → run() → 反量化 → softmax → argmax
     *
     * @param jpg_data  指向 JPEG 文件字节流的指针（通常在 rodata 中）
     * @param jpg_len   JPEG 数据长度（字节）
     * @return classification_result_t 分类结果
     */
    classification_result_t infer(const uint8_t *jpg_data, size_t jpg_len);

    /**
     * @brief 直接使用 PC 预处理后的 INT8 数据进行推理（跳过预处理，只测模型）
     *
     * @param int8_data  指向 [224x224x3] INT8 NHWC 数据的指针
     * @return classification_result_t 分类结果
     */
    classification_result_t infer_from_preprocessed(const int8_t *int8_data);

    /**
     * @brief 运行诊断测试
     *
     * 使用三种已知输入模式验证模型：
     *   - 全 0
     *   - 全 127（最大正 INT8）
     *   - 渐变值
     *
     * 如果三种模式输出不同，则模型和权重加载正常；
     * 如果输出相同，说明权重未正确加载。
     */
    void run_diagnostic();

private:
    void *m_model;          /**< 指向 dl::Model 的不透明指针 */
    void *m_preprocessor;   /**< 指向 dl::image::ImagePreprocessor (仅 NHWC 模式) */
    bool  m_initialized;    /**< init() 是否已成功调用 */
    bool  m_is_nchw;        /**< true=NCHW 手动量化, false=NHWC ImagePreprocessor */
};
