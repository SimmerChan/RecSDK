#include <cmath>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "attention_fusion_grad_tiling.h"

namespace optiling {

#define RESERVER_UB_SIZE (20 * 1024)

int CeilDiv(int a, int b)
{
    return (a + b - 1) / b;
}

static int32_t GradMatmulTile(gert::TilingContext* context, AttentionFusionGradTilingData &tilingData)
{
    // q (B, M, K) k (B, N, K) v (B, V, K)
    auto qShape = context->GetInputShape(2)->GetStorageShape();
    auto kShape = context->GetInputShape(3)->GetStorageShape();
    auto vShape = context->GetInputShape(4)->GetStorageShape();

    auto ascnedPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    // qkMatmul configuration
    matmul_tiling::MatmulApiTiling gardV(ascnedPlatform);
    gardV.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardV.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardV.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardV.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardV.SetShape(kShape.GetDim(1), vShape.GetDim(2), qShape.GetDim(1));

    gardV.SetBias(false);
    gardV.SetBufferSpace(-1, 100*1024, -1);

    matmul_tiling::MatmulApiTiling gardS(ascnedPlatform);
    gardS.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardS.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardS.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardS.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardS.SetShape(qShape.GetDim(1), vShape.GetDim(1), vShape.GetDim(2));

    gardS.SetBias(false);
    gardS.SetBufferSpace(-1, -1, -1);

    matmul_tiling::MatmulApiTiling gardQ(ascnedPlatform);
    gardQ.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardQ.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardQ.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardQ.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardQ.SetShape(qShape.GetDim(1), qShape.GetDim(2), kShape.GetDim(1));

    gardQ.SetBias(false);
    gardQ.SetBufferSpace(-1, -1, -1);

    matmul_tiling::MatmulApiTiling gardK(ascnedPlatform);
    gardK.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardK.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardK.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardK.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardK.SetShape(kShape.GetDim(1), qShape.GetDim(2), qShape.GetDim(1));

    gardK.SetBias(false);
    gardK.SetBufferSpace(-1, -1, -1);

    // Get tilingData using on the kernel side 
    if (gardV.GetTiling(tilingData.gardVMatmulTiling) == -1 ||
        gardS.GetTiling(tilingData.gardSMatmulTiling) == -1 ||
        gardK.GetTiling(tilingData.gardKMatmulTiling) == -1 ||
        gardQ.GetTiling(tilingData.gardQMatmulTiling) == -1) {
        return 1;
    }
    return 0;
}

static int32_t GradSoftmaxTiling(gert::TilingContext* context, AttentionFusionGradTilingData &tilingData, uint64_t ub)
{
    // q (B, M, K) k (B, N, K) v (B, V, K)
    auto qShape = context->GetInputShape(2)->GetStorageShape();
    auto kShape = context->GetInputShape(3)->GetStorageShape();
    auto vShape = context->GetInputShape(4)->GetStorageShape();
    
    // std::vector<int64_t> softmaxShape = 
    int paddingKeyDim1 = CeilDiv(kShape.GetDim(1), 8) * 8;
    // 3*2+1
    int numRowOfNormalizeOne = ub / 4 / sizeof(float) / paddingKeyDim1 / 8 * 8;
    int keyDim1Align;
    if (kShape.GetDim(1) % 8 == 0) {
        keyDim1Align = 1;
    } else if (kShape.GetDim(1) == 50 && qShape.GetDim(1) > 500) {
        numRowOfNormalizeOne = 16 * 8;
        keyDim1Align = 2;
    } else {
        keyDim1Align = 3;
    }

    const ge::Shape softmaxShape({numRowOfNormalizeOne, paddingKeyDim1});
    const uint32_t localWorkSpaceSize = AscendC::GetSoftMaxGradMaxTmpSize(softmaxShape, sizeof(float), false, false);
    float attenDimSqrt = 1/std::sqrt(qShape.GetDim(2));

    // set attr
    tilingData.set_attnDim(qShape.GetDim(2));
    tilingData.set_queryDim1(qShape.GetDim(1));
    tilingData.set_queryDim2(qShape.GetDim(2));
    tilingData.set_keyDim1(kShape.GetDim(1));
    tilingData.set_keyDim2(kShape.GetDim(2));
    tilingData.set_valueDim1(vShape.GetDim(1));
    tilingData.set_valueDim2(vShape.GetDim(2));
    tilingData.set_batchNum(qShape.GetDim(0));
    tilingData.set_numRowOfNormalnizeOne(numRowOfNormalizeOne);
    tilingData.set_paddingKeyDim1(paddingKeyDim1);
    tilingData.set_attenDimSqrt(attenDimSqrt);
    tilingData.set_keyDim1Align(keyDim1Align);

    AscendC::SoftMaxGradTilingFunc(softmaxShape, sizeof(float), localWorkSpaceSize, tilingData.softMaxGradTiling);
    return 0;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AttentionFusionGradTilingData tilingData;
    // Platform configuration
    auto ascnedPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int softmaxOutSize = qShape.GetDim(0) * qShape.GetDim(1) * kShape.GetDim(1)*sizeof(float);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    size_t systemWorkspacesSize = ascnedPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = softmaxOutSize + systemWorkspacesSize;
    size_t coreNum = ascnedPlatform.GetCoreNumAic();
    
    uint64_t ub;
    ascnedPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    ub = ub - RESERVER_UB_SIZE;

    if (GradMatmulTiling(context, tilingData) != 0 || GradSoftmaxTiling(context, tilingData, ub) != 0) {
        return ge::GRAPH_FAILED;
    }

    std::vector<int64_t> shapeVec = {16, 50 * 8};
    ge::Shape srcShape(shapeVec);
    AscendC::GetConfusionTransposeTilingInfo(srcShape, 0, sizeof(float), 7, tilingData.confusionTransposeTilingData);

    std::vector<int64_t> shapeVec1 = {56 * 8, 16};
    ge::Shape srcShape1(shapeVec1);
    AscendC::GetConfusionTransposeTilingInfo(srcShape1, 0, sizeof(float), 7, tilingData.confusionTransposeTilingData1);

    std::vector<int64_t> shapeVec2 = {16, 56 * 8};
    ge::Shape srcShape2(shapeVec2);
    AscendC::GetConfusionTransposeTilingInfo(srcShape2, 0, sizeof(float), 7, tilingData.confusionTransposeTilingData2);

    std::vector<int64_t> shapeVec3 = {50* 8, 16};
    ge::Shape srcShape3(shapeVec3);
    AscendC::GetConfusionTransposeTilingInfo(srcShape3, 0, sizeof(float), 7, tilingData.confusionTransposeTilingData3);
   
    context->SetBlockDim(coreNum);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* qShape = context->GetInputShape(2);
    const gert::Shape* kShape = context->GetInputShape(3);
    const gert::Shape* vShape = context->GetInputShape(4);
    gert::Shape* gradQShape = context->GetOutputShape(0);
    gert::Shape* gradKShape = context->GetOutputShape(1);
    gert::Shape* gradVShape = context->GetOutputShape(2);
    *gradQShape = *qShape;
    *gradKShape = *kShape;
    *gradVShape = *vShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(2));
    context->SetOutputDataType(1, context->GetInputDataType(3));
    context->SetOutputDataType(2, context->GetInputDataType(4));
    return GRAPH_SUCCESS;
}
}


namespace ops {
class AttentionFusionGrad : public OpDef {
public:
    explicit AttentionFusionGrad(const char* name) : OpDef(name)
    {
        this->Input("dout")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("softmax_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("key")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("value")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("grad_query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("grad_key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("grad_value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDtype);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910");

    }
};

OP_ADD(AttentionFusionGrad);
}
