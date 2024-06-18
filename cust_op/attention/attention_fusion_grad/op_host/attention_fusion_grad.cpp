
#include "attention_fusion_grad_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <cmath>

constexpr int RESERVER_UB_SIZE = 20 * 1024;
namespace optiling {

int CeilDiv(int a, int b) {
    return (a+b-1)/b;
}


static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    std::cout << "testing start " << std::endl;
    // q (B, M, K) k (B, N, K) v (B, V, K)
    auto qShape = context->GetInputShape(2)->GetStorageShape();
    auto kShape = context->GetInputShape(3)->GetStorageShape();
    auto vShape = context->GetInputShape(4)->GetStorageShape();

    // Platform configuration
    int softmaxOutSize = qShape.GetDim(0) * qShape.GetDim(1) * kShape.GetDim(1)*sizeof(float);
    auto ascnedPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    size_t systemWorkspacesSize = ascnedPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = softmaxOutSize + systemWorkspacesSize;
    size_t coreNum = ascnedPlatform.GetCoreNumAic();


    // qkMatmul configuration
    matmul_tiling::MatmulApiTiling gardVMatmulTiling(ascnedPlatform);
    gardVMatmulTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardVMatmulTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardVMatmulTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardVMatmulTiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardVMatmulTiling.SetShape(kShape.GetDim(1), vShape.GetDim(2), qShape.GetDim(1));
    // qkMatmulTiling.SetBatchInfoForNormal(qShape.GetDim(0), qShape.GetDim(0), qShape.GetDim(1), kShape.GetDim(1), kShape.GetDim(2));
    //gardVMatmulTiling.SetSingleShape(kShape.GetDim(1), vShape.GetDim(2), qShape.GetDim(1));
    std::cout << "gardVMatmulTiling shape " << " " << kShape.GetDim(1) << " " << vShape.GetDim(2) << " " << qShape.GetDim(1) << std::endl;
    // qkMatmulTiling.SetBatchNum(1);
    gardVMatmulTiling.SetBias(false);
    gardVMatmulTiling.SetBufferSpace(-1, 100*1024, -1);
    // gardVMatmulTiling.SetDim(coreNum);

    matmul_tiling::MatmulApiTiling gardSMatmulTiling(ascnedPlatform);
    gardSMatmulTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardSMatmulTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardSMatmulTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardSMatmulTiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardSMatmulTiling.SetShape(qShape.GetDim(1), vShape.GetDim(1), vShape.GetDim(2));
    // qkMatmulTiling.SetBatchInfoForNormal(qShape.GetDim(0), qShape.GetDim(0), qShape.GetDim(1), kShape.GetDim(1), kShape.GetDim(2));
    // gardSMatmulTiling.SetSingleShape(qShape.GetDim(1), vShape.GetDim(1), vShape.GetDim(2));
    std::cout << "gardSMatmulTiling shape " << " " << qShape.GetDim(1) << " " << vShape.GetDim(1) << " " << vShape.GetDim(2) << std::endl;
    // qkMatmulTiling.SetBatchNum(1);
    gardSMatmulTiling.SetBias(false);
    gardSMatmulTiling.SetBufferSpace(-1, -1, -1);
    // gardSMatmulTiling.SetDim(coreNum);

    matmul_tiling::MatmulApiTiling gardQMatmulTiling(ascnedPlatform);
    gardQMatmulTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardQMatmulTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardQMatmulTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardQMatmulTiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardQMatmulTiling.SetShape(qShape.GetDim(1), qShape.GetDim(2), kShape.GetDim(1));
    // qkMatmulTiling.SetBatchInfoForNormal(qShape.GetDim(0), qShape.GetDim(0), qShape.GetDim(1), kShape.GetDim(1), kShape.GetDim(2));
    // gardQMatmulTiling.SetSingleShape(qShape.GetDim(1), qShape.GetDim(2), kShape.GetDim(1));
    std::cout << "gardQMatmulTiling shape " << " " << qShape.GetDim(1) << " " << qShape.GetDim(2)<< " " << kShape.GetDim(1) << std::endl;
    // qkMatmulTiling.SetBatchNum(1);
    gardQMatmulTiling.SetBias(false);
    gardQMatmulTiling.SetBufferSpace(-1, -1, -1);
    // gardQMatmulTiling.SetDim(coreNum);

    matmul_tiling::MatmulApiTiling gardKMatmulTiling(ascnedPlatform);
    gardKMatmulTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardKMatmulTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardKMatmulTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardKMatmulTiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    gardKMatmulTiling.SetShape(kShape.GetDim(1), qShape.GetDim(2), qShape.GetDim(1));
    // qkMatmulTiling.SetBatchInfoForNormal(qShape.GetDim(0), qShape.GetDim(0), qShape.GetDim(1), kShape.GetDim(1), kShape.GetDim(2));
    // gardKMatmulTiling.SetSingleShape(kShape.GetDim(1), qShape.GetDim(2), qShape.GetDim(1));
    std::cout << "gardKMatmulTiling shape " << " " << kShape.GetDim(1) << " " << qShape.GetDim(2)<< " " << qShape.GetDim(1) << std::endl;
    // qkMatmulTiling.SetBatchNum(1);
    gardKMatmulTiling.SetBias(false);
    gardKMatmulTiling.SetBufferSpace(-1, -1, -1);
    // gardKMatmulTiling.SetDim(coreNum);
    // VC Parallel size
    // kvBmm Matmul Tilling
    // matmul_tiling::MultiCoreMatmulTiling kvMatmulTiling(ascnedPlatform);
    // kvMatmulTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    // kvMatmulTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    // kvMatmulTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    // kvMatmulTiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    // kvMatmulTiling.SetShape(qShape.GetDim(1), vShape.GetDim(2), kShape.GetDim(1));
    // kvMatmulTiling.SetSingleShape(qShape.GetDim(1), vShape.GetDim(2), kShape.GetDim(1));
    // std::cout << "kvMatmulTiling shape " << " " << qShape.GetDim(1) << " " << vShape.GetDim(2) << " " << kShape.GetDim(1) << std::endl;;
    // kvMatmulTiling.SetBias(false);
    // kvMatmulTiling.SetBufferSpace(-1, -1, -1);
    // kvMatmulTiling.SetDim(coreNum);

    // softmax tiling
    // std::vector<int64_t> softmaxShape = 
    int paddingKeyDim1 = CeilDiv(kShape.GetDim(1), 8)*8;
    uint64_t ub;
    ascnedPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    ub = ub - RESERVER_UB_SIZE;
    // 3*2+1
    int numRowOfNormalnizeOne = ub/4/sizeof(float)/paddingKeyDim1/8*8;
    int keyDim1Align;
    if (kShape.GetDim(1)%8 == 0) {
        keyDim1Align = 1;
    } else if (kShape.GetDim(1) == 50 && qShape.GetDim(1)>500) {
        numRowOfNormalnizeOne = 16*8;
        keyDim1Align = 2;
    } else {
        keyDim1Align = 3;
    }
    
    
    std::cout << "numOfNormalnizeOne  keyDim1PadinngSize" << " " << numRowOfNormalnizeOne << " " << paddingKeyDim1 <<  std::endl;

    const ge::Shape softmaxShape({numRowOfNormalnizeOne, paddingKeyDim1});
    const uint32_t localWorkSpaceSize = AscendC::GetSoftMaxGradMaxTmpSize(softmaxShape, sizeof(float), false, false);
    

    // set attr
    AttentionFusionGradTilingData tilingData;
    tilingData.set_attnDim(qShape.GetDim(2));
    tilingData.set_queryDim1(qShape.GetDim(1));
    tilingData.set_queryDim2(qShape.GetDim(2));
    tilingData.set_keyDim1(kShape.GetDim(1));
    tilingData.set_keyDim2(kShape.GetDim(2));
    tilingData.set_valueDim1(vShape.GetDim(1));
    tilingData.set_valueDim2(vShape.GetDim(2));
    tilingData.set_batchNum(qShape.GetDim(0));
    tilingData.set_numRowOfNormalnizeOne(numRowOfNormalnizeOne);
    tilingData.set_paddingKeyDim1(paddingKeyDim1);
    float attenDimSqrt = 1/std::sqrt(qShape.GetDim(2));
    tilingData.set_attenDimSqrt(attenDimSqrt);
    tilingData.set_keyDim1Align(keyDim1Align);

    // Get tilingData using on the kernel side 
    if (gardVMatmulTiling.GetTiling(tilingData.gardVMatmulTiling)) {
        return ge::GRAPH_FAILED;
    }
    if (gardSMatmulTiling.GetTiling(tilingData.gardSMatmulTiling)) {
        return ge::GRAPH_FAILED;
    }
    if (gardKMatmulTiling.GetTiling(tilingData.gardKMatmulTiling)) {
        return ge::GRAPH_FAILED;
    }
    if (gardQMatmulTiling.GetTiling(tilingData.gardQMatmulTiling)) {
        return ge::GRAPH_FAILED;
    }
    // if (kvMatmulTiling.GetTiling(tilingData.kvMatmulTiling)) {
    //     return ge::GRAPH_FAILED;
    // }
    AscendC::SoftMaxGradTilingFunc(softmaxShape, sizeof(float), localWorkSpaceSize, tilingData.softMaxGradTiling);

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
