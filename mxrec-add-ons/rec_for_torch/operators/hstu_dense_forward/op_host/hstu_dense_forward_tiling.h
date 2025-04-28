#ifndef HSTU_DENSE_FORWARD_TILING_H
#define HSTU_DENSE_FORWARD_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling_policy_define.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(HstuDenseForwardTilingData)
TILING_DATA_FIELD_DEF(uint32_t, size);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, qkMatmul);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, svMatmul);

TILING_DATA_FIELD_DEF(int64_t, batchSize);
TILING_DATA_FIELD_DEF(int64_t, seqLen);
TILING_DATA_FIELD_DEF(int64_t, headNum);
TILING_DATA_FIELD_DEF(int64_t, dim);
TILING_DATA_FIELD_DEF(int64_t, blockHeight);

TILING_DATA_FIELD_DEF(int32_t, qkBaseM);
TILING_DATA_FIELD_DEF(int32_t, qkBaseN);
TILING_DATA_FIELD_DEF(int32_t, svBaseM);
TILING_DATA_FIELD_DEF(int32_t, svBaseN);

#ifdef SUPPORT_V200
    TILING_DATA_FIELD_DEF(int32_t, tmpUbSize);
#else
    TILING_DATA_FIELD_DEF_ARR(uint32_t, (HstuDenseForward::MAX_BATCH_SIZE + 1), seqOffset);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, HstuDenseForward::MAX_AIV_NUM, eachCoreStartBlockId);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, HstuDenseForward::MAX_AIV_NUM, eachCoreEndBlockId);
#endif

TILING_DATA_FIELD_DEF(uint32_t, enableBias);
TILING_DATA_FIELD_DEF(uint32_t, maskType);
TILING_DATA_FIELD_DEF(int64_t, maxSeqLen);
TILING_DATA_FIELD_DEF(float, siluScale);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(HstuDenseForward, HstuDenseForwardTilingData)
}  // namespace optiling
#endif