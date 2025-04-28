#ifndef HSTU_DENSE_BACKWARD_TILING_H
#define HSTU_DENSE_BACKWARD_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

#include "hstu_dense_backward_tiling_common.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(HstuDenseBackwardTilingData)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, qkMatmul);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, qGradMatmul);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, kGradMatmul);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, vGradMatmul);

TILING_DATA_FIELD_DEF(int64_t, batchSize);
TILING_DATA_FIELD_DEF(int64_t, seqLen);
TILING_DATA_FIELD_DEF(int64_t, headNum);
TILING_DATA_FIELD_DEF(int64_t, headDim);

TILING_DATA_FIELD_DEF(int64_t, blockHeight);
TILING_DATA_FIELD_DEF(int64_t, dataTypeLength);

TILING_DATA_FIELD_DEF_ARR(uint32_t, (MAX_BATCH_SIZE + 1), seqOffset);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_AIV_NUM, eachCoreStartColBlockId);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_AIV_NUM, eachCoreEndColBlockId);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_AIV_NUM, eachCoreStartRowBlockId);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_AIV_NUM, eachCoreEndRowBlockId);

TILING_DATA_FIELD_DEF(int32_t, maskType);
TILING_DATA_FIELD_DEF(int32_t, enableBias);
TILING_DATA_FIELD_DEF(int32_t, maxSeqLen);
TILING_DATA_FIELD_DEF(int32_t, biasGradSeqLen);
TILING_DATA_FIELD_DEF(float, siluScale);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(HstuDenseBackward, HstuDenseBackwardTilingData)
} // namespace optiling
#endif