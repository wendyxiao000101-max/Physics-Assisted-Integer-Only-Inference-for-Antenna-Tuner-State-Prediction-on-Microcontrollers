#include "fp32_runtime.h"

#include <math.h>
#include <string.h>

#define MODEL_BYTES 11144
#define HIDDEN 32
#define CLASSES 15

typedef struct {
    int inputs;
    int outputs;
    const float *weights;
    const float *bias;
} Linear;

static Linear stem;
static Linear block_fc1;
static Linear block_fc2;
static Linear head;
static int loaded;

static float stem_weights[5 * HIDDEN];
static float stem_bias[HIDDEN];
static float block1_weights[HIDDEN * HIDDEN];
static float block1_bias[HIDDEN];
static float block2_weights[HIDDEN * HIDDEN];
static float block2_bias[HIDDEN];
static float head_weights[HIDDEN * CLASSES];

static void take_floats(const uint8_t **cursor, float *destination, size_t count)
{
    memcpy(destination, *cursor, count * sizeof(float));
    *cursor += count * sizeof(float);
}

int fp32_model_load(const uint8_t *blob, size_t len)
{
    loaded = 0;
    if (blob == NULL || len != MODEL_BYTES || memcmp(blob, "F32R32V1", 8) != 0) {
        return -1;
    }

    const uint8_t *cursor = blob + 8;
    take_floats(&cursor, stem_weights, 5 * HIDDEN);
    take_floats(&cursor, stem_bias, HIDDEN);
    take_floats(&cursor, block1_weights, HIDDEN * HIDDEN);
    take_floats(&cursor, block1_bias, HIDDEN);
    take_floats(&cursor, block2_weights, HIDDEN * HIDDEN);
    take_floats(&cursor, block2_bias, HIDDEN);
    take_floats(&cursor, head_weights, HIDDEN * CLASSES);
    stem = (Linear){5, HIDDEN, stem_weights, stem_bias};
    block_fc1 = (Linear){HIDDEN, HIDDEN, block1_weights, block1_bias};
    block_fc2 = (Linear){HIDDEN, HIDDEN, block2_weights, block2_bias};
    head = (Linear){HIDDEN, CLASSES, head_weights, NULL};
    if (cursor != blob + len) {
        return -2;
    }
    loaded = 1;
    return 0;
}

static void linear(const Linear *layer, const float *input, float *output)
{
    for (int row = 0; row < layer->outputs; row++) {
        float value = layer->bias == NULL ? 0.0f : layer->bias[row];
        const float *weights = layer->weights + row * layer->inputs;
        for (int col = 0; col < layer->inputs; col++) {
            value += input[col] * weights[col];
        }
        output[row] = value;
    }
}

static float gelu(float value)
{
    return 0.5f * value * (1.0f + erff(value * 0.70710678118654752440f));
}

int fp32_predict(const float features[5])
{
    float hidden[HIDDEN];
    float block[HIDDEN];
    float residual[HIDDEN];
    float scores[CLASSES];

    if (!loaded || features == NULL) {
        return -1;
    }

    linear(&stem, features, hidden);
    for (int i = 0; i < HIDDEN; i++) {
        hidden[i] = gelu(hidden[i]);
    }

    linear(&block_fc1, hidden, block);
    for (int i = 0; i < HIDDEN; i++) {
        if (block[i] < 0.0f) {
            block[i] = 0.0f;
        }
    }

    linear(&block_fc2, block, residual);
    for (int i = 0; i < HIDDEN; i++) {
        residual[i] += hidden[i];
        if (residual[i] < 0.0f) {
            residual[i] = 0.0f;
        }
    }

    linear(&head, residual, scores);
    int best = 0;
    scores[0] *= 20.0f;
    for (int i = 1; i < CLASSES; i++) {
        scores[i] *= 20.0f;
        if (scores[i] > scores[best]) {
            best = i;
        }
    }
    return best;
}
