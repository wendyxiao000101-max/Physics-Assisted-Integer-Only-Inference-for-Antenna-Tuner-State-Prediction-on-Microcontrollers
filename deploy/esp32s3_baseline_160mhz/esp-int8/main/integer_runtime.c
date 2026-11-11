/* Bit-exact with repository integer_runtime.py (I8R32V02). */
#include "integer_runtime.h"

#include <string.h>

#define MODEL_BYTES 4656
#define LUT_N 1024

typedef struct {
    int ni;
    int no;
    const int8_t *w;
    const int32_t *b;
    const int32_t *m;
} Layer;

static int32_t offset[5];
static int32_t step[5];
static int32_t hz;
static int32_t skip;
static int8_t lut[LUT_N];
static Layer layers[4];
static int8_t w0[5 * 32], w1[32 * 32], w2[32 * 32], w3[32 * 15];
static int32_t b0[32], m0[32], b1[32], m1[32], b2[32], m2[32], b3[15], m3[15];
static int loaded;

static int32_t rd_i32(const uint8_t **p)
{
    int32_t v;
    memcpy(&v, *p, 4);
    *p += 4;
    return v;
}

static void rd_i8s(const uint8_t **p, int8_t *dst, int n)
{
    memcpy(dst, *p, (size_t)n);
    *p += n;
}

static void rd_i32s(const uint8_t **p, int32_t *dst, int n)
{
    for (int i = 0; i < n; i++) {
        dst[i] = rd_i32(p);
    }
}

int int8_model_load(const uint8_t *blob, size_t len)
{
    loaded = 0;
    if (blob == NULL || len != MODEL_BYTES || memcmp(blob, "I8R32V02", 8) != 0) {
        return -1;
    }
    const uint8_t *p = blob + 8;
    rd_i32s(&p, offset, 5);
    rd_i32s(&p, step, 5);
    hz = rd_i32(&p);
    rd_i8s(&p, lut, LUT_N);

    const int shapes[4][2] = {{5, 32}, {32, 32}, {32, 32}, {32, 15}};
    int8_t *ws[4] = {w0, w1, w2, w3};
    int32_t *bs[4] = {b0, b1, b2, b3};
    int32_t *ms[4] = {m0, m1, m2, m3};
    for (int L = 0; L < 4; L++) {
        int ni = shapes[L][0];
        int no = shapes[L][1];
        rd_i8s(&p, ws[L], ni * no);
        rd_i32s(&p, bs[L], no);
        rd_i32s(&p, ms[L], no);
        layers[L] = (Layer){ni, no, ws[L], bs[L], ms[L]};
    }
    skip = rd_i32(&p);
    if (p != blob + MODEL_BYTES) {
        return -2;
    }
    for (int i = 0; i < 5; i++) {
        if (step[i] == 0) {
            return -3;
        }
    }
    loaded = 1;
    return 0;
}

static int32_t clipi(int32_t v, int32_t lo, int32_t hi)
{
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

static int32_t rq(int64_t v)
{
    return (int32_t)((v + ((int64_t)1 << 23)) >> 24);
}

/* Python integer // (floor), not C toward-zero division. */
static int64_t floor_div(int64_t a, int64_t b)
{
    int64_t q = a / b;
    int64_t r = a % b;
    if (r != 0 && ((a < 0 && b > 0) || (a > 0 && b < 0))) {
        q--;
    }
    return q;
}

static void linear(const int8_t *q, int L, int32_t zp, int64_t *out)
{
    const Layer *ly = &layers[L];
    for (int j = 0; j < ly->no; j++) {
        int32_t acc = ly->b[j];
        const int8_t *row = ly->w + j * ly->ni;
        for (int i = 0; i < ly->ni; i++) {
            acc += (int32_t)(q[i] - zp) * (int32_t)row[i];
        }
        out[j] = (int64_t)acc * (int64_t)ly->m[j];
    }
}

int int8_predict(const int32_t feat[5])
{
    int8_t x[5], h[32], a[32], z[32];
    int64_t acc[32];

    if (!loaded || feat == NULL) {
        return -1;
    }

    for (int i = 0; i < 5; i++) {
        int64_t num = 2 * ((int64_t)feat[i] - offset[i]) + step[i];
        int64_t den = 2 * (int64_t)step[i];
        int32_t u = clipi((int32_t)floor_div(num, den), 0, 255);
        x[i] = (int8_t)(u - 128);
    }

    linear(x, 0, -128, acc);
    for (int j = 0; j < 32; j++) {
        h[j] = lut[clipi(rq(acc[j]), -512, 511) + 512];
    }

    linear(h, 1, hz - 128, acc);
    for (int j = 0; j < 32; j++) {
        a[j] = (int8_t)(clipi(rq(acc[j]), 0, 255) - 128);
    }

    linear(a, 2, -128, acc);
    for (int j = 0; j < 32; j++) {
        int64_t t = acc[j] + ((int64_t)h[j] - hz + 128) * (int64_t)skip;
        z[j] = (int8_t)(clipi(rq(t), 0, 255) - 128);
    }

    linear(z, 3, -128, acc);
    int best = 0;
    int32_t bests = rq(acc[0]);
    for (int j = 1; j < 15; j++) {
        int32_t s = rq(acc[j]);
        if (s > bests) {
            bests = s;
            best = j;
        }
    }
    return best;
}
