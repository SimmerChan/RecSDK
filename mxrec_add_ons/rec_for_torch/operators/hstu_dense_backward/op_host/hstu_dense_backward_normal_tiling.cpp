#include "register/op_def_registry.h"

#include "hstu_dense_backward_normal_tiling.h"

namespace optiling {
ge::graphStatus GetNormalAttrsInfo(const gert::RuntimeAttrs *attrs, HstuDenseBackwardTilingData &tiling)
{
    const int32_t *maskType = attrs->GetAttrPointer<int32_t>(INDEX_T::INDEX_1);
    OPS_LOGD_IF_NULL(maskType, return ge::GRAPH_FAILED);
    tiling.set_maskType(*maskType);

    const int32_t *maxSeqLen = attrs->GetAttrPointer<int32_t>(INDEX_T::INDEX_2);
    OPS_LOGD_IF_NULL(maxSeqLen, return ge::GRAPH_FAILED);
    tiling.set_maxSeqLen(*maxSeqLen);

    const float *siluScale = attrs->GetAttrPointer<float>(INDEX_T::INDEX_3);
    OPS_LOGD_IF_NULL(siluScale, return ge::GRAPH_FAILED);
    tiling.set_siluScale(*siluScale);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetNormalBasicShapeInfo(gert::TilingContext *context, HstuDenseBackwardTilingData &tiling)
{
    int32_t maxSeqLen = tiling.get_maxSeqLen();
    auto gradShape = context->GetInputShape(INDEX_T::INDEX_0)->GetStorageShape();
    auto attnBiasGradShape = context->GetOutputShape(INDEX_T::INDEX_3)->GetStorageShape();
    OPS_LOG_D_IF(gradShape.GetDimNum() != GRAD_DIM_NUM, return ge::GRAPH_FAILED,
                 "hstu normal backward only support input with dim %d\n");

    int64_t batchSize = gradShape.GetDim(INDEX_T::INDEX_0);
    int64_t seqLen = gradShape.GetDim(INDEX_T::INDEX_1);
    int64_t headNum = gradShape.GetDim(INDEX_T::INDEX_2);
    int64_t headDim = gradShape.GetDim(INDEX_T::INDEX_3);
    int32_t biasGradSeqLen = attnBiasGradShape.GetDim(INDEX_T::INDEX_2);

    OPS_LOG_D_IF(biasGradSeqLen < maxSeqLen, return ge::GRAPH_FAILED, "attnBiasGrad get seqLen less than maxSeqLen\n");

    tiling.set_batchSize(batchSize);
    tiling.set_seqLen(seqLen);
    tiling.set_headNum(headNum);
    tiling.set_headDim(headDim);
    tiling.set_biasGradSeqLen(biasGradSeqLen);

    OPS_LOG_D_IF(!BasicShapeCheck(batchSize, seqLen, headNum, headDim), return ge::GRAPH_FAILED,
                 "normal shape check failed\n");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckMaskTypeAndBias(gert::TilingContext *context,
                                     HstuDenseBackwardTilingData &tiling)
{
    auto batchSize = tiling.get_batchSize();
    auto headNum = tiling.get_headNum();
    auto maxSeqLen = tiling.get_maxSeqLen();
    auto maskType = tiling.get_maskType();

    auto attnBiasGradShape = context->GetOutputShape(INDEX_T::INDEX_3)->GetStorageShape();

    auto attnBias = context->GetOptionalInputTensor(INDEX_T::INDEX_5);
    if (attnBias == nullptr) {
        tiling.set_enableBias(0);
    } else {
        tiling.set_enableBias(1);

        auto attnBiasShape = context->GetInputShape(INDEX_T::INDEX_5)->GetStorageShape();
        OPS_LOG_D_IF(!IsSameShape(attnBiasShape, attnBiasGradShape, BIAS_DIM_NUM), return ge::GRAPH_FAILED,
                     "attnBias shape not equal with attnBiasGrad\n");
    }

    if (IfMask(maskType, MaskType::MASK_CUSTOM)) {
        auto mask = context->GetOptionalInputTensor(INDEX_T::INDEX_4);
        OPS_LOG_IF_NULL(mask, return ge::GRAPH_FAILED);

        auto maskShape = context->GetInputShape(INDEX_T::INDEX_4)->GetStorageShape();
        OPS_LOG_D_IF(maskShape.GetDimNum() != MASK_DIM_NUM, return ge::GRAPH_FAILED,
            "mask dim num is not %d\n", MASK_DIM_NUM);

        OPS_LOG_D_IF(maskShape.GetDim(INDEX_T::INDEX_0) != batchSize ||
                     maskShape.GetDim(INDEX_T::INDEX_1) != headNum ||
                     maskShape.GetDim(INDEX_T::INDEX_2) != maxSeqLen ||
                     maskShape.GetDim(INDEX_T::INDEX_3) != maxSeqLen,
                     return ge::GRAPH_FAILED, "mask shape must be {batchSize, headNum, seqLen, seqLen}\n");
    } else if (IfMask(maskType, MaskType::MASK_TRIL) ||
               IfMask(maskType, MaskType::MASK_NONE)) {
        // do nothing
    } else if (IfMask(maskType, MaskType::MASK_TRIU)) {
        OPS_LOG_D("maskType:MASK_TRIU is not support yet\n");
        return ge::GRAPH_FAILED;
    } else {
        OPS_LOG_D("supported maskType list is [MASK_TRIL, MASK_NONE, MASK_CUSTOM]\n");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InitNormalTilingKey(gert::TilingContext *context, HstuDenseBackwardTilingData &tiling)
{
    int64_t dataTypeLength = 0;
    ge::DataType gradType = context->GetInputTensor(INDEX_T::INDEX_0)->GetDataType();
    if (gradType == ge::DataType::DT_FLOAT) {
        dataTypeLength = DATA_TYPE_LENGTH_FLOAT;
        context->SetTilingKey(FLOAT_TILING_KEY);
        tiling.set_blockHeight(BLOCK_128);
    } else if (gradType == ge::DataType::DT_FLOAT16) {
        dataTypeLength = DATA_TYPE_LENGTH_FLOAT16;
        context->SetTilingKey(FLOAT16_TILING_KEY);
        tiling.set_blockHeight(BLOCK_256);
    } else if (gradType == ge::DataType::DT_BF16) {
        dataTypeLength = DATA_TYPE_LENGTH_FLOAT16;
        context->SetTilingKey(BF16_TILING_KEY);
        tiling.set_blockHeight(BLOCK_256);
    } else {
        OPS_LOG_D("invalid datatype, only support float/fp16/bf16");
        return ge::GRAPH_FAILED;
    }
    tiling.set_dataTypeLength(dataTypeLength);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingNormalFunc(gert::TilingContext *context,
                                 const gert::RuntimeAttrs *attrs,
                                 HstuDenseBackwardTilingData &tiling)
{
    OPS_LOG_D_IF(GetNormalAttrsInfo(attrs, tiling) == ge::GRAPH_FAILED, return ge::GRAPH_FAILED,
                 "NormalTiling GetNormalAttrsInfo failed\n");

    OPS_LOG_D_IF(GetNormalBasicShapeInfo(context, tiling) == ge::GRAPH_FAILED, return ge::GRAPH_FAILED,
                 "NormalTiling GetNormalBasicShapeInfo failed\n");

    OPS_LOG_D_IF(CheckMaskTypeAndBias(context, tiling) == ge::GRAPH_FAILED, return ge::GRAPH_FAILED,
                 "NormalTiling CheckMaskTypeAndBias failed\n");

    OPS_LOG_D_IF(InitNormalTilingKey(context, tiling) == ge::GRAPH_FAILED, return ge::GRAPH_FAILED,
                 "NormalTiling InitNormalTilingKey failed\n");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormalInferShape(gert::InferShapeContext *context)
{
    const gert::Shape *qShape = context->GetInputShape(INDEX_T::INDEX_1);
    OPS_LOGD_IF_NULL(qShape, return ge::GRAPH_FAILED);

    // q_grad、k_grad、v_grad的shape与q一致
    gert::Shape *qGradShape = context->GetOutputShape(INDEX_T::INDEX_0);
    OPS_LOGD_IF_NULL(qGradShape, return ge::GRAPH_FAILED);
    qGradShape->SetDimNum(qShape->GetDimNum());

    gert::Shape *kGradShape = context->GetOutputShape(INDEX_T::INDEX_1);
    OPS_LOGD_IF_NULL(kGradShape, return ge::GRAPH_FAILED);
    kGradShape->SetDimNum(qShape->GetDimNum());

    gert::Shape *vGradShape = context->GetOutputShape(INDEX_T::INDEX_2);
    OPS_LOGD_IF_NULL(vGradShape, return ge::GRAPH_FAILED);
    vGradShape->SetDimNum(qShape->GetDimNum());

    for (size_t i = 0; i < qShape->GetDimNum(); i++) {
        qGradShape->SetDim(i, qShape->GetDim(i));
        kGradShape->SetDim(i, qShape->GetDim(i));
        vGradShape->SetDim(i, qShape->GetDim(i));
    }

    gert::Shape *attnBiasGradShape = context->GetOutputShape(INDEX_T::INDEX_3);
    OPS_LOGD_IF_NULL(attnBiasGradShape, return ge::GRAPH_FAILED);
    attnBiasGradShape->SetDimNum(BIAS_DIM_NUM);
    attnBiasGradShape->SetDim(INDEX_T::INDEX_0, qShape->GetDim(INDEX_T::INDEX_0));
    attnBiasGradShape->SetDim(INDEX_T::INDEX_1, qShape->GetDim(INDEX_T::INDEX_2));
    attnBiasGradShape->SetDim(INDEX_T::INDEX_2, qShape->GetDim(INDEX_T::INDEX_1));
    attnBiasGradShape->SetDim(INDEX_T::INDEX_3, qShape->GetDim(INDEX_T::INDEX_1));

    return ge::GRAPH_SUCCESS;
}
}