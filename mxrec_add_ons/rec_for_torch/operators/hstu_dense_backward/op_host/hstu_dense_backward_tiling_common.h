#ifndef HSTU_DENSE_BACKWARD_TILING_COMMON_H
#define HSTU_DENSE_BACKWARD_TILING_COMMON_H

#include <cstdint>
#include <cstdio>

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

constexpr int MAX_BATCH_SIZE = 2048;
constexpr int MAX_SEQ_LENS = 20480;

constexpr int JAGGED_FLOAT_TILING_KEY = 5;
constexpr int JAGGED_BF16_TILING_KEY = 4;
constexpr int JAGGED_FLOAT16_TILING_KEY = 3;
constexpr int FLOAT_TILING_KEY = 2;
constexpr int BF16_TILING_KEY = 1;
constexpr int FLOAT16_TILING_KEY = 0;
constexpr int MAX_AIV_NUM = 48;

constexpr int GRAD_DIM_NUM = 4;
constexpr int JAGGED_GRAD_DIM_NUM = 3;
constexpr int MASK_DIM_NUM = 4;
constexpr int BIAS_DIM_NUM = 4;
constexpr int MID_USE_TIMES = 2;

constexpr int DATA_TYPE_FLOAT = 32;
constexpr int DATA_TYPE_FLOAT16 = 16;
constexpr int DATA_TYPE_LENGTH_FLOAT = 4;
constexpr int DATA_TYPE_LENGTH_FLOAT16 = 2;

constexpr int BLOCK_128 = 128;
constexpr int BLOCK_256 = 256;

constexpr int VCORE_NUM_IN_ONE_AIC = 2;

enum class InputLayout { NORMAL = 0, JAGGED = 1 };
enum class MaskType { MASK_TRIL = 0, MASK_TRIU = 1, MASK_NONE = 2, MASK_CUSTOM = 3 };

namespace INDEX_T {
constexpr int INDEX_0 = 0;
constexpr int INDEX_1 = 1;
constexpr int INDEX_2 = 2;
constexpr int INDEX_3 = 3;
constexpr int INDEX_4 = 4;
constexpr int INDEX_5 = 5;
} // namespace INDEX_T

#ifndef OPS_LOGD_IF_NULL
#define OPS_LOGD_IF_NULL(PTR, EXPR)                                            \
    if (__builtin_expect((PTR) == nullptr, 0)) {                               \
        printf("%s is nullptr\n", #PTR);                                       \
        EXPR;                                                                  \
    }
#endif

#ifndef OPS_LOGD
#define OPS_LOGD(FMT, ...)                                                     \
    do {                                                                       \
        printf(FMT, ##__VA_ARGS__);                                            \
    } while (0);
#endif

#ifndef OPS_LOGD_IF
#define OPS_LOGD_IF(COND, LOG_FUNC, EXPR)                                      \
    static_assert(std::is_same<bool, std::decay<decltype(COND)>::type>::value, \
                  "condition should be bool");                                 \
    do {                                                                       \
        if (__builtin_expect((COND), 0)) {                                     \
            LOG_FUNC;                                                          \
            EXPR;                                                              \
        }                                                                      \
    } while (0);
#endif

struct ShapeRange {
public:
    int64_t lbound {0}; // shape下限
    int64_t ubound {0}; // shape上限
    int64_t mutiple {0}; // 倍数
    const char *name {nullptr};
    ShapeRange(int64_t lbound, int64_t ubound, int64_t mutiple, const char *name);
    bool Check(int64_t val) const;
};

ge::graphStatus GetInputLayout(const gert::RuntimeAttrs *attrs, InputLayout &layout);

bool IfMask(const int32_t &maskType, MaskType maskTypeEnum);

bool IsSameShape(const gert::Shape &shape0, const gert::Shape &shape1, int dim);

bool BasicShapeCheck(int64_t batchSize, int64_t seqLen, int64_t headNum, int64_t dim);

#endif