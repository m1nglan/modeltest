/**
 * @file main.cpp
 * @brief ESP-DL 模型测试主程序
 *
 * 功能：
 *   1. 初始化螺丝/垫圈二分类模型 (ClassifierDriver)
 *   2. 运行诊断测试（用已知输入模式验证模型工作正常）
 *   3. 对 t1.jpg / t2.jpg 分别推理并输出分类结果
 *
 * 硬件平台：ESP32-S3 (N16R8: 16MB Flash + 8MB Octal PSRAM)
 * 框架：     ESP-DL (esp-dl) + ESP-IDF v6.0.1
 * 模型：     MobileNetV2 风格 INT8 量化二分类器
 *           输入 [3x224x224] NCHW, 输出 [2] INT8
 * 类别：     0 = screw (螺丝), 1 = washer (垫圈)
 */

#include <stdio.h>
#include "esp_log.h"
#include "driver/classifier_driver.hpp"

/* 日志标签 */
static const char *TAG = "modeltest";

/*
 * JPG 图片通过 CMakeLists.txt 中的 target_add_aligned_binary_data()
 * 嵌入到固件的 .rodata 段。链接器自动生成 _binary_*_start / _binary_*_end 符号。
 * 符号命名规则：_binary_<文件名>_start，文件名中的点变为下划线。
 * 例如 t1.jpg → _binary_t1_jpg_start / _binary_t1_jpg_end
 */
extern const uint8_t t1_jpg_start[] asm("_binary_t1_jpg_start");
extern const uint8_t t1_jpg_end[]   asm("_binary_t1_jpg_end");
extern const uint8_t t2_jpg_start[] asm("_binary_t2_jpg_start");
extern const uint8_t t2_jpg_end[]   asm("_binary_t2_jpg_end");
extern const uint8_t t3_jpg_start[] asm("_binary_t3_jpg_start");
extern const uint8_t t3_jpg_end[]   asm("_binary_t3_jpg_end");
extern const uint8_t t4_jpg_start[] asm("_binary_t4_jpg_start");
extern const uint8_t t4_jpg_end[]   asm("_binary_t4_jpg_end");

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "ESP-DL Model Test: screw/washer classifier");

    /*
     * 步骤 1：初始化分类驱动
     * - 从 rodata 加载模型 (model.espdl)
     * - 读取输入张量信息（形状 [3,224,224], 类型 INT8, exponent=-5）
     */
    ClassifierDriver driver;
    esp_err_t ret = driver.init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize classifier driver");
        return;
    }

    /*
     * 步骤 2：测试 t1.jpg（垫圈）
     * 预期输出：washer (class_id=0)
     */
    {
        size_t len = (size_t)(t1_jpg_end - t1_jpg_start);
        classification_result_t res = driver.infer(t1_jpg_start, len);
        ESP_LOGI(TAG, "t1.jpg => %s(%d) score=%.4f", res.label, res.class_id, res.score);
    }

    /*
     * 步骤 4：测试 t2.jpg
     * 预期输出：class=screw (class_id=1)
     */
    {
        size_t len = (size_t)(t2_jpg_end - t2_jpg_start);
        classification_result_t res = driver.infer(t2_jpg_start, len);
        ESP_LOGI(TAG, "t2.jpg => %s(%d) score=%.4f", res.label, res.class_id, res.score);
    }

    /* 步骤 5：t3.jpg（垫圈） */
    {
        size_t len = (size_t)(t3_jpg_end - t3_jpg_start);
        classification_result_t res = driver.infer(t3_jpg_start, len);
        ESP_LOGI(TAG, "t3.jpg => %s(%d) score=%.4f", res.label, res.class_id, res.score);
    }

    /* 步骤 6：t4.jpg（螺丝） */
    {
        size_t len = (size_t)(t4_jpg_end - t4_jpg_start);
        classification_result_t res = driver.infer(t4_jpg_start, len);
        ESP_LOGI(TAG, "t4.jpg => %s(%d) score=%.4f", res.label, res.class_id, res.score);
    }

    ESP_LOGI(TAG, "All tests complete.");
}
