#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int int8_model_load(const uint8_t *blob, size_t len);
int int8_predict(const int32_t feat[5]);

#ifdef __cplusplus
}
#endif
