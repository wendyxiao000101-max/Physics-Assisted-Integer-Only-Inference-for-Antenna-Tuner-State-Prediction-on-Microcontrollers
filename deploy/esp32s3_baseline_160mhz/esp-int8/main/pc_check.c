/* Host check: gcc -O2 -o /tmp/int8_pc_check pc_check.c integer_runtime.c model_blob.c */
#include "integer_runtime.h"
#include <stdio.h>

extern const uint8_t model_int8_blob[];
extern const size_t model_int8_blob_len;

int main(void)
{
    if (int8_model_load(model_int8_blob, model_int8_blob_len) != 0) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    const int32_t samples[][5] = {
        {9021, -3716, 737, 531, 18969},
        {-1332, 20179, 846, -6596, 1067},
        {16274, 21437, 845, 13931, 19855},
        {27634, -366, 707, 15107, -12810},
        {9788, 21216, 746, 9043, 20304},
        {-14402, -16027, 703, -12643, -16027},
        {-30000, -30000, 700, -30000, -30000},
    };
    for (unsigned i = 0; i < sizeof(samples) / sizeof(samples[0]); i++) {
        printf("%d\n", int8_predict(samples[i]));
    }
    return 0;
}
