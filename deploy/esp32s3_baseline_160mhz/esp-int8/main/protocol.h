#pragma once

#include <stdint.h>

#define HOST_MAGIC0 0xA5u
#define HOST_MAGIC1 0x5Au
#define DEV_MAGIC0  0x5Au
#define DEV_MAGIC1  0xA5u

#define CMD_PING    0x00u
#define CMD_PREDICT 0x01u

#define RSP_PONG    0x80u
#define RSP_RESULT  0x81u
#define RSP_ERR     0x8Fu

#define MODEL_IDENT_LEN 8
#define SHA256_LEN 32
#define FEATURE_DIM 5
