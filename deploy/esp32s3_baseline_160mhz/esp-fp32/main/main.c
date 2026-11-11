#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "esp_cpu.h"
#include "esp_partition.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "fp32_runtime.h"

#define DATASET_ROWS 96768
#define FEATURES 5
#define DATASET_BYTES ((size_t)DATASET_ROWS * FEATURES * sizeof(int16_t) + DATASET_ROWS)
#define WARMUP_ROWS 100
#define CPU_MHZ 160
#define HIST_BUCKET_CYCLES 64
#define HIST_BUCKETS 16384

extern const uint8_t fp32_model_start[] asm("_binary_fp32_model_bin_start");
extern const uint8_t fp32_model_end[] asm("_binary_fp32_model_bin_end");

static const double feature_mean[FEATURES] = {
    13965.23264922, 10802.05065724, 765.71775794, 2255.51790881, 8177.99197049,
};
static const double feature_scale[FEATURES] = {
    10544.15344313, 11311.47302761, 54.57597201, 8229.86924800, 10971.56968030,
};

static uint32_t latency_hist[HIST_BUCKETS];

static void normalize(const int16_t *raw, float output[FEATURES])
{
    for (int i = 0; i < FEATURES; i++) {
        output[i] = (float)(((double)raw[i] - feature_mean[i]) / feature_scale[i]);
    }
}

static uint32_t counter_overhead(void)
{
    uint32_t best = UINT32_MAX;
    for (int i = 0; i < 1000; i++) {
        uint32_t start = esp_cpu_get_cycle_count();
        uint32_t elapsed = esp_cpu_get_cycle_count() - start;
        if (elapsed < best) {
            best = elapsed;
        }
    }
    return best;
}

static uint32_t percentile_cycles(uint32_t rank)
{
    uint32_t seen = 0;
    for (uint32_t i = 0; i < HIST_BUCKETS; i++) {
        seen += latency_hist[i];
        if (seen >= rank) {
            return (i + 1) * HIST_BUCKET_CYCLES - 1;
        }
    }
    return UINT32_MAX;
}

static void print_us(const char *name, uint64_t milli_us)
{
    printf(" %s=%llu.%03llu",
           name,
           (unsigned long long)(milli_us / 1000),
           (unsigned long long)(milli_us % 1000));
}

static void report(uint32_t correct,
                   uint64_t total_cycles,
                   uint32_t min_cycles,
                   uint32_t p50_cycles,
                   uint32_t p95_cycles,
                   uint32_t p99_cycles,
                   uint32_t max_cycles,
                   uint32_t overhead_cycles,
                   uint32_t overflow)
{
    uint32_t errors = DATASET_ROWS - correct;
    uint64_t scaled = (uint64_t)correct * 100000000ull / DATASET_ROWS;
    printf("RESULT rows=%u correct=%lu errors=%lu accuracy=%u.%06u inference_total_us=%lu\n",
           DATASET_ROWS,
           (unsigned long)correct,
           (unsigned long)errors,
           (unsigned)(scaled / 1000000ull),
           (unsigned)(scaled % 1000000ull),
           (unsigned long)(total_cycles / CPU_MHZ));
    printf("LATENCY_CYCLES pure_inference count=%u overhead=%lu min=%lu mean=%llu"
           " p50=%lu p95=%lu p99=%lu max=%lu histogram_overflow=%lu\n",
           DATASET_ROWS,
           (unsigned long)overhead_cycles,
           (unsigned long)min_cycles,
           (unsigned long long)(total_cycles / DATASET_ROWS),
           (unsigned long)p50_cycles,
           (unsigned long)p95_cycles,
           (unsigned long)p99_cycles,
           (unsigned long)max_cycles,
           (unsigned long)overflow);
    printf("LATENCY_US pure_inference");
    print_us("min", (uint64_t)min_cycles * 1000 / CPU_MHZ);
    print_us("mean", total_cycles * 1000 / ((uint64_t)DATASET_ROWS * CPU_MHZ));
    print_us("p50", (uint64_t)p50_cycles * 1000 / CPU_MHZ);
    print_us("p95", (uint64_t)p95_cycles * 1000 / CPU_MHZ);
    print_us("p99", (uint64_t)p99_cycles * 1000 / CPU_MHZ);
    print_us("max", (uint64_t)max_cycles * 1000 / CPU_MHZ);
    printf("\n");
    fflush(stdout);
}

void app_main(void)
{
    printf("BOOT embedded-dataset fp32\n");
    fflush(stdout);

    const esp_partition_t *part =
        esp_partition_find_first(ESP_PARTITION_TYPE_DATA, 0x40, "dataset");
    if (part == NULL || part->size < DATASET_BYTES) {
        printf("ERR dataset partition\n");
        return;
    }

    const void *mapped = NULL;
    esp_partition_mmap_handle_t handle = 0;
    if (esp_partition_mmap(part, 0, DATASET_BYTES,
                           ESP_PARTITION_MMAP_DATA, &mapped, &handle) != ESP_OK) {
        printf("ERR dataset mmap\n");
        return;
    }

    size_t model_len = (size_t)(fp32_model_end - fp32_model_start);
    if (fp32_model_load(fp32_model_start, model_len) != 0) {
        printf("ERR model_load bytes=%u\n", (unsigned)model_len);
        return;
    }

    const uint8_t *blob = mapped;
    const int16_t *features = (const int16_t *)blob;
    const uint8_t *labels = blob + (size_t)DATASET_ROWS * FEATURES * sizeof(int16_t);

    volatile int warmup_sink = 0;
    for (uint32_t i = 0; i < WARMUP_ROWS; i++) {
        float input[FEATURES];
        normalize(features + (size_t)i * FEATURES, input);
        warmup_sink += fp32_predict(input);
    }

    uint32_t overhead = counter_overhead();
    printf("START rows=%u warmup=%u cpu_mhz=%u timer_overhead_cycles=%lu sink=%d\n",
           DATASET_ROWS,
           WARMUP_ROWS,
           CPU_MHZ,
           (unsigned long)overhead,
           warmup_sink);
    fflush(stdout);

    uint32_t correct = 0;
    uint32_t min_cycles = UINT32_MAX;
    uint32_t max_cycles = 0;
    uint32_t overflow = 0;
    uint64_t total_cycles = 0;

    for (uint32_t i = 0; i < DATASET_ROWS; i++) {
        int16_t raw[FEATURES];
        float input[FEATURES];
        const int16_t *row = features + (size_t)i * FEATURES;
        for (int k = 0; k < FEATURES; k++) {
            raw[k] = row[k];
        }

        uint32_t start = esp_cpu_get_cycle_count();
        normalize(raw, input);
        int prediction = fp32_predict(input);
        uint32_t cycles = esp_cpu_get_cycle_count() - start;
        cycles = cycles > overhead ? cycles - overhead : 0;

        total_cycles += cycles;
        if (cycles < min_cycles) {
            min_cycles = cycles;
        }
        if (cycles > max_cycles) {
            max_cycles = cycles;
        }
        uint32_t bucket = cycles / HIST_BUCKET_CYCLES;
        if (bucket < HIST_BUCKETS) {
            latency_hist[bucket]++;
        } else {
            overflow++;
        }
        if (prediction == (int)labels[i]) {
            correct++;
        }

        if ((i + 1) % 4096u == 0) {
            vTaskDelay(1);
        }
    }

    uint32_t rank50 = (DATASET_ROWS * 50u + 99u) / 100u;
    uint32_t rank95 = (DATASET_ROWS * 95u + 99u) / 100u;
    uint32_t rank99 = (DATASET_ROWS * 99u + 99u) / 100u;
    uint32_t measured = DATASET_ROWS - overflow;
    uint32_t p50 = measured < rank50 ? max_cycles : percentile_cycles(rank50);
    uint32_t p95 = measured < rank95 ? max_cycles : percentile_cycles(rank95);
    uint32_t p99 = measured < rank99 ? max_cycles : percentile_cycles(rank99);
    if (p50 > max_cycles) {
        p50 = max_cycles;
    }
    if (p95 > max_cycles) {
        p95 = max_cycles;
    }
    if (p99 > max_cycles) {
        p99 = max_cycles;
    }

    report(correct, total_cycles, min_cycles, p50, p95, p99,
           max_cycles, overhead, overflow);
    for (;;) {
        vTaskDelay(portMAX_DELAY);
    }
}
