#include "kernel_operator.h"
#include "attention_fusion_grad_kernel.h"
#include "utils.h"
#ifdef __CCE_KT_TEST__
#include "attention_fusion_grad_tiling.h"
#endif

using namespace AscendC;

extern "C" __global__ __aicore__ void attention_fusion_grad(GM_ADDR dout, GM_ADDR softmaxOut, GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR gradQery, GM_ADDR gradKey, GM_ADDR gradValue, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    const TCubeTiling* gardVMatmulTiling = &tiling_data.gardVMatmulTiling;
    const TCubeTiling* gardSMatmulTiling = &tiling_data.gardSMatmulTiling;
    const TCubeTiling* gardQMatmulTiling = &tiling_data.gardQMatmulTiling;
    const TCubeTiling* gardKMatmulTiling = &tiling_data.gardKMatmulTiling;
    const SoftMaxTiling* softMaxGradTiling = &tiling_data.softMaxGradTiling;

    AttentionFusionGradArgs attentionFusionGradAgs {
        dout, softmaxOut, query, key, value, gradQery, gradKey, gradValue, workspace,
        tiling_data.queryDim1, tiling_data.queryDim2, tiling_data.keyDim1, tiling_data.keyDim2,
        tiling_data.valueDim1, tiling_data.valueDim2, tiling_data.batchNum, tiling_data.numRowOfNormalizeOne, tiling_data.paddingKeyDim1, tiling_data.attenDimSqrt, 
        tiling_data.keyDim1Align, gardVMatmulTiling, gardSMatmulTiling, gardQMatmulTiling, gardKMatmulTiling, softMaxGradTiling, &tiling_data.confusionTransposeTilingData, &tiling_data.confusionTransposeTilingData1, 
        &tiling_data.confusionTransposeTilingData2,  &tiling_data.confusionTransposeTilingData3 
    };
    AttentionFusionGradKernel<float> attentionGradKernel;
    attentionGradKernel.Compute(attentionFusionGradAgs);

}