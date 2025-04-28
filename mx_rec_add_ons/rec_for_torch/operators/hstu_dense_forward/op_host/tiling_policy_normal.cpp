#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling_policy_factory.h"
#include "tiling_policy_normal.h"

namespace HstuDenseForward {

REGISTER_POLICY(LAYOUT_TYPE::NORMAL, std::make_shared<TilingPolicyNormal>());

bool TilingPolicyNormal::TilingShape(gert::TilingContext* context, optiling::HstuDenseForwardTilingData &tiling)
{
    auto qShape = context->GetInputShape(0)->GetStorageShape();

    int64_t batchSize = qShape.GetDim(0);
    tiling.set_batchSize(batchSize);
    int64_t seqLen = qShape.GetDim(1);
    tiling.set_seqLen(seqLen);
    int64_t headNum = qShape.GetDim(2);
    tiling.set_headNum(headNum);
    int64_t dim = qShape.GetDim(3);
    tiling.set_dim(dim);

    OPS_LOGD_IF(!GeneralShapeCheck(batchSize, seqLen, headNum, dim), printf("Shape Check failed"), return false);
    return true;
}

bool TilingPolicyNormal::TilingKeySet(gert::TilingContext* context, optiling::HstuDenseForwardTilingData &tiling)
{
    ge::DataType qTypeGe = context->GetInputTensor(0)->GetDataType();
    if (qTypeGe == ge::DataType::DT_FLOAT) {
        context->SetTilingKey(FLOAT_TILING_KEY);
    } else if (qTypeGe == ge::DataType::DT_FLOAT16) {
        context->SetTilingKey(FLOAT16_TILING_KEY);
    } else if (qTypeGe == ge::DataType::DT_BF16) {
        context->SetTilingKey(BF16_TILING_KEY);
    } else {
        printf("invalid datatype, only support fp32, fp16, bf16.\n");
        return false;
    }

    return true;
}

}