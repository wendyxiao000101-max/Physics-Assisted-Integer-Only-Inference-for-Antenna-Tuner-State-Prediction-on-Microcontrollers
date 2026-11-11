#include "fp32_runtime.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define ROWS 96768
#define FEATURES 5
#define DATASET_BYTES ((size_t)ROWS * FEATURES * sizeof(int16_t) + ROWS)

static const double feature_mean[FEATURES] = {
    13965.23264922, 10802.05065724, 765.71775794, 2255.51790881, 8177.99197049,
};
static const double feature_scale[FEATURES] = {
    10544.15344313, 11311.47302761, 54.57597201, 8229.86924800, 10971.56968030,
};

static uint8_t *read_file(const char *path, size_t *length)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    rewind(file);
    uint8_t *data = malloc((size_t)size);
    if (data == NULL || fread(data, 1, (size_t)size, file) != (size_t)size) {
        free(data);
        fclose(file);
        return NULL;
    }
    fclose(file);
    *length = (size_t)size;
    return data;
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: %s MODEL DATASET REFERENCE_PREDICTIONS\n", argv[0]);
        return 2;
    }

    size_t model_size = 0;
    size_t dataset_size = 0;
    uint8_t *model = read_file(argv[1], &model_size);
    uint8_t *dataset = read_file(argv[2], &dataset_size);
    FILE *reference = fopen(argv[3], "r");
    if (model == NULL || dataset == NULL || reference == NULL ||
        dataset_size != DATASET_BYTES || fp32_model_load(model, model_size) != 0) {
        fprintf(stderr, "input or model load failed\n");
        return 1;
    }

    char line[256];
    fgets(line, sizeof(line), reference);
    const int16_t *features = (const int16_t *)dataset;
    const uint8_t *labels = dataset + (size_t)ROWS * FEATURES * sizeof(int16_t);
    uint32_t correct = 0;
    uint32_t agreement = 0;

    for (uint32_t row = 0; row < ROWS; row++) {
        float input[FEATURES];
        for (int i = 0; i < FEATURES; i++) {
            input[i] = (float)(((double)features[(size_t)row * FEATURES + i] -
                                feature_mean[i]) /
                               feature_scale[i]);
        }
        int prediction = fp32_predict(input);
        correct += prediction == labels[row];

        unsigned index;
        int expected;
        if (fgets(line, sizeof(line), reference) == NULL ||
            sscanf(line, "%u,%d", &index, &expected) != 2 || index != row) {
            fprintf(stderr, "reference parse failed at row %u\n", row);
            return 1;
        }
        agreement += prediction == expected;
    }

    printf("rows=%u correct=%u agreement=%u\n", ROWS, correct, agreement);
    fclose(reference);
    free(dataset);
    free(model);
    return agreement == ROWS ? 0 : 1;
}
