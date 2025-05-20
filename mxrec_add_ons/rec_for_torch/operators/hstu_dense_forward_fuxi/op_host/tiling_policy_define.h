#ifndef TILING_POLICY_DEFINE_H
#define TILING_POLICY_DEFINE_H

namespace HstuDenseForwardFuxi {

#include <cstdio>
#include <cstdint>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

#ifndef OPS_LOGD_IF_NULL
#define OPS_LOGD_IF_NULL(PTR, EXPR)                                     \
    if (__builtin_expect((PTR) == nullptr, 0)) {                        \
        printf("%s is nullptr!", #PTR);                                 \
        EXPR;                                                           \
    }
#endif

#ifndef OPS_LOGD
#define OPS_LOGD(FMT, ...)                                              \
    do {                                                                \
        printf(FMT, ##__VA_ARGS__);                                     \
    } while (0)
#endif

#define OPS_LOGD_IF(COND, LOG_FUNC, EXPR)                               \
    static_assert(std::is_same<bool, std::decay<decltype(COND)>::type>::value, "condition should be bool"); \
    do {                                                                \
        if (__builtin_expect((COND), 0)) {                              \
            LOG_FUNC;                                                   \
            EXPR;                                                       \
        }                                                               \
    } while (0)

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

constexpr int MAX_AIV_NUM = 48;
constexpr int MAX_BATCH_SIZE = 10;
constexpr int BLOCK_HEIGHT = 64;
constexpr int VCORE_NUM_IN_ONE_AIC = 1;
constexpr int COMPUTE_PIPE_NUM = 3;
constexpr int TRANS_PIPE_NUM = 4;
constexpr int TRANS_TASK_NUM = 3;

constexpr int OUTPUT_DIM_NUM = 3;

}

#endif