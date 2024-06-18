#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
namespace optiling {
    BEGIN_TILING_DATA_DEF(AttentionFusionTilingData)
        TILING_DATA_FIELD_DEF(uint8_t, normalizeAttr);
        TILING_DATA_FIELD_DEF(float, attnDim);
        TILING_DATA_FIELD_DEF(int32_t, queryDim1);
        TILING_DATA_FIELD_DEF(int32_t, queryDim2);
        TILING_DATA_FIELD_DEF(int32_t, keyDim1);
        TILING_DATA_FIELD_DEF(int32_t, keyDim2);
        TILING_DATA_FIELD_DEF(int32_t, valueDim1);
        TILING_DATA_FIELD_DEF(int32_t, valueDim2);
        TILING_DATA_FIELD_DEF(int32_t, batchNum);
        TILING_DATA_FIELD_DEF(int32_t, normalizeLoop);
        TILING_DATA_FIELD_DEF(int32_t, normalizeRow);
        TILING_DATA_FIELD_DEF(int32_t, normalizeColumn);
        TILING_DATA_FIELD_DEF(float, normalizeSqrt);
        TILING_DATA_FIELD_DEF(uint64_t, maxSharedTmpBuf);
        TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, qkMatmulTiling);
        TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, kvMatmulTiling);
        TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softMaxTilingData);
        TILING_DATA_FIELD_DEF_STRUCT(ConfusionTransposeTiling, confusionTransposeTilingData);
        TILING_DATA_FIELD_DEF_STRUCT(ConfusionTransposeTiling, confusionTransposeTilingData1);
        TILING_DATA_FIELD_DEF_STRUCT(ConfusionTransposeTiling, confusionTransposeTilingData2);
        TILING_DATA_FIELD_DEF_STRUCT(ConfusionTransposeTiling, confusionTransposeTilingData3);
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(AttentionFusion, AttentionFusionTilingData)
}
