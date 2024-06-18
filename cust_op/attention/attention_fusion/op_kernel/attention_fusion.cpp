#include "kernel_operator.h"
#include "attention_fusion_kernel.h"
using namespace AscendC;

// call of kernel function
extern "C" __global__ __aicore__ void attention_fusion(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attnMask, GM_ADDR attenScore, GM_ADDR softmaxOut, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    const TCubeTiling *qkMatmulTiling = &tiling_data.qkMatmulTiling;
    const TCubeTiling *kvMatmulTiling = &tiling_data.kvMatmulTiling;
    const SoftMaxTiling *softMaxTilingData = &tiling_data.softMaxTilingData;
    // TODO: user kernel impl
    AttentionFusionArgs args {
        query, key, value, attnMask, attenScore, softmaxOut, tiling_data.normalizeAttr, tiling_data.queryDim1,
        tiling_data.queryDim2, tiling_data.keyDim1, tiling_data.keyDim2, tiling_data.valueDim1, tiling_data.valueDim2,
        tiling_data.batchNum, tiling_data.normalizeLoop, tiling_data.normalizeRow, tiling_data.normalizeColumn,
        tiling_data.normalizeSqrt, tiling_data.maxSharedTmpBuf, qkMatmulTiling,
        kvMatmulTiling, softMaxTilingData, &tiling_data.confusionTransposeTilingData, &tiling_data.confusionTransposeTilingData1, 
        &tiling_data.confusionTransposeTilingData2,  &tiling_data.confusionTransposeTilingData3 
    };

    AttentionFusionKernel<float, float, float> kernel;
    kernel.Compute(args);
}