#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int fp32_model_load(const uint8_t *blob, size_t len);
int fp32_predict(const float features[5]);

#ifdef __cplusplus
}
#endif
