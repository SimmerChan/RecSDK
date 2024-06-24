#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AttentionFusionGradTilingData)
        TILING_DATA_FIELD_DEF(float, attnDim);
        TILING_DATA_FIELD_DEF(int32_t, queryDim1);
        TILING_DATA_FIELD_DEF(int32_t, queryDim2);
        TILING_DATA_FIELD_DEF(int32_t, keyDim1);
        TILING_DATA_FIELD_DEF(int32_t, keyDim2);
        TILING_DATA_FIELD_DEF(int32_t, valueDim1);
        TILING_DATA_FIELD_DEF(int32_t, valueDim2);
        TILING_DATA_FIELD_DEF(int32_t, batchNum);
        TILING_DATA_FIELD_DEF(int32_t, numRowOfNormalizeOne);
        TILING_DATA_FIELD_DEF(int32_t, paddingKeyDim1);
        TILING_DATA_FIELD_DEF(float, attenDimSqrt);
        TILING_DATA_FIELD_DEF(int32_t, keyDim1Align);
        TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, gardVMatmulTiling);
        TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, gardSMatmulTiling);
        TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, gardQMatmulTiling);
        TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, gardKMatmulTiling);
        TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softMaxGradTiling);

        TILING_DATA_FIELD_DEF_STRUCT(ConfusionTransposeTiling, confusionTransposeTilingData);
        TILING_DATA_FIELD_DEF_STRUCT(ConfusionTransposeTiling, confusionTransposeTilingData1);
        TILING_DATA_FIELD_DEF_STRUCT(ConfusionTransposeTiling, confusionTransposeTilingData2);
        TILING_DATA_FIELD_DEF_STRUCT(ConfusionTransposeTiling, confusionTransposeTilingData3);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AttentionFusionGrad, AttentionFusionGradTilingData)
}
