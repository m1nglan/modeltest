#pragma once

#include <cstdint>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Classification result for a single image
 */
typedef struct {
    int class_id;        /**< 0 = screw, 1 = washer */
    float score;         /**< Raw dequantized logit for the predicted class */
    float probability;   /**< Softmax probability for the predicted class */
    const char *label;   /**< "screw" or "washer" */
} classification_result_t;

#ifdef __cplusplus
}
#endif

/**
 * @brief ESP-DL based classifier driver for screw/washer 2-class model
 *
 * Handles model loading, JPEG decoding, image preprocessing,
 * inference, and output dequantization/postprocessing.
 */
class ClassifierDriver {
public:
    ClassifierDriver();
    ~ClassifierDriver();

    /**
     * @brief Initialize the classifier: load model from flash partition,
     *        create image preprocessor.
     * @return ESP_OK on success, ESP_FAIL otherwise.
     */
    esp_err_t init();

    /**
     * @brief Run inference on a JPEG image.
     * @param jpg_data  Pointer to JPEG file data in memory.
     * @param jpg_len   Length of JPEG data in bytes.
     * @return Classification result (class_id, score, probability, label).
     */
    classification_result_t infer(const uint8_t *jpg_data, size_t jpg_len);

private:
    void *m_model;          /**< Opaque pointer to dl::Model */
    bool  m_initialized;
};
