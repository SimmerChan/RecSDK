#ifndef TILING_POLICY_DEFINE_H
#define TILING_POLICY_DEFINE_H

namespace HstuDenseForward {

#include <cstdio>
#include <cstdint>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "../../../common/ops_log.h"

namespace INDEX_T {
    constexpr int INDEX_0 = 0;
    constexpr int INDEX_1 = 1;
    constexpr int INDEX_2 = 2;
    constexpr int INDEX_3 = 3;
    constexpr int INDEX_4 = 4;
    constexpr int INDEX_5 = 5;
}

constexpr int FLOAT_TILING_KEY = 2;
constexpr int BF16_TILING_KEY = 1;
constexpr int FLOAT16_TILING_KEY = 0;

constexpr int JAGGED_TILING_KEY_OFFSET = 3;
constexpr int JAGGED_FLOAT16_TILING_KEY = FLOAT16_TILING_KEY + JAGGED_TILING_KEY_OFFSET;
constexpr int JAGGED_FLOAT_TILING_KEY = FLOAT_TILING_KEY + JAGGED_TILING_KEY_OFFSET;
constexpr int JAGGED_BF16_TILING_KEY = BF16_TILING_KEY + JAGGED_TILING_KEY_OFFSET;

constexpr int MAX_AIV_NUM = 48;
constexpr int MAX_BATCH_SIZE = 2048;
#ifdef SUPPORT_V200
    constexpr int BLOCK_HEIGHT = 128;
#else
    constexpr int BLOCK_HEIGHT = 256;
#endif
constexpr int VCORE_NUM_IN_ONE_AIC = 2;
constexpr int COMPUTE_PIPE_NUM = 3;
constexpr int TRANS_PIPE_NUM = 4;
constexpr int TRANS_TASK_NUM = 3;

}

#endif