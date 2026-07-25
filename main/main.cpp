#include <stdio.h>
#include "esp_log.h"
#include "driver/classifier_driver.hpp"

static const char *TAG = "modeltest";

/* JPEG binary symbols embedded via target_add_aligned_binary_data */
extern const uint8_t t1_jpg_start[] asm("_binary_t1_jpg_start");
extern const uint8_t t1_jpg_end[]   asm("_binary_t1_jpg_end");
extern const uint8_t t2_jpg_start[] asm("_binary_t2_jpg_start");
extern const uint8_t t2_jpg_end[]   asm("_binary_t2_jpg_end");

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "ESP-DL Model Test: screw/washer classifier");

    /* Initialize the classifier driver */
    ClassifierDriver driver;
    esp_err_t ret = driver.init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize classifier driver");
        return;
    }

    /* Test t1.jpg */
    {
        size_t len = (size_t)(t1_jpg_end - t1_jpg_start);
        ESP_LOGI(TAG, "--- Processing t1.jpg (%zu bytes) ---", len);
        classification_result_t res = driver.infer(t1_jpg_start, len);
        ESP_LOGI(TAG, "t1.jpg => class=%s (%d), score=%.4f, prob=%.4f%%",
                 res.label, res.class_id, res.score, res.probability * 100.0f);
    }

    /* Test t2.jpg */
    {
        size_t len = (size_t)(t2_jpg_end - t2_jpg_start);
        ESP_LOGI(TAG, "--- Processing t2.jpg (%zu bytes) ---", len);
        classification_result_t res = driver.infer(t2_jpg_start, len);
        ESP_LOGI(TAG, "t2.jpg => class=%s (%d), score=%.4f, prob=%.4f%%",
                 res.label, res.class_id, res.score, res.probability * 100.0f);
    }

    ESP_LOGI(TAG, "All tests complete.");
}
