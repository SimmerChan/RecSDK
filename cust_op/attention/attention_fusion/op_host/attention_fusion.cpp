#include <cstdint>
#include <cmath>
#include "attention_fusion_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
// using namespace matmul_tiling;

#define TEST_LOG(fmt, args...) fprintf(stdout, fmt "\n", ##args)

namespace optiling {

#define RESERVER_UB_SIZE (20 * 1024)

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    std::cout << "testing start " << std::endl;
    // Platform configuration
    auto ascnedPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    size_t systemWorkspacesSize = ascnedPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = 0 + systemWorkspacesSize;
    size_t coreNum = ascnedPlatform.GetCoreNumAic();

    // q (B, M, K) k (B, N, K) v (B, V, K)
    auto qShape = context->GetInputShape(0)->GetStorageShape();
    auto kShape = context->GetInputShape(1)->GetStorageShape();
    auto vShape = context->GetInputShape(2)->GetStorageShape();

    // qkMatmul configuration
    matmul_tiling::MultiCoreMatmulTiling qkMatmulTiling(ascnedPlatform);
    qkMatmulTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    qkMatmulTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    qkMatmulTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    qkMatmulTiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    qkMatmulTiling.SetShape(qShape.GetDim(1), kShape.GetDim(1), kShape.GetDim(2));
    // qkMatmulTiling.SetBatchInfoForNormal(qShape.GetDim(0), qShape.GetDim(0), qShape.GetDim(1), kShape.GetDim(1), kShape.GetDim(2));
    qkMatmulTiling.SetSingleShape(qShape.GetDim(1), kShape.GetDim(1), kShape.GetDim(2));
    std::cout << "qkMatmulTiling shape " << " " << qShape.GetDim(1) << " " << kShape.GetDim(1) << " " << kShape.GetDim(2) << std::endl;
    // qkMatmulTiling.SetBatchNum(1);
    qkMatmulTiling.SetBias(false);
    qkMatmulTiling.SetBufferSpace(-1, -1, -1);
    qkMatmulTiling.SetDim(coreNum);

    // VC Parallel size
    uint64_t ub;
    ascnedPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    ub = ub - RESERVER_UB_SIZE;
    int numOfelement = kShape.GetDim(1) / 8;
    uint8_t attr = 0;
    int normalizeColumn = 0;
    /* Get column with 32bytes alignment*/
    if ((kShape.GetDim(1) % 8) == 0) {
        normalizeColumn = numOfelement * 8;
        attr = 1;
    } else if (kShape.GetDim(1) == 50 && qShape.GetDim(1)>500) {
        normalizeColumn = numOfelement * 8 + 8;
        attr = 2;
    } else {
        normalizeColumn = numOfelement * 8 + 8;
    }

    if ((sizeof(float) * normalizeColumn) > (ub / 3)) {
        TEST_LOG("[ERROR] key dim1 too large, LocalWorkSize insufficient");
        return ge::GRAPH_FAILED; 
    }

    /* Get how many rows that half of ub contains */
    int normalizeRow = ub / 3 / (sizeof(float) * normalizeColumn);
    if (normalizeRow > qShape.GetDim(1)) {
        normalizeRow = qShape.GetDim(1);
    }

    if (attr == 2) {
        normalizeRow = 16*16;
    }

    /* Get max Ub left for softmax shared tmp buffer */
    uint64_t maxLocalWorkSize = ub - (normalizeRow * normalizeColumn * sizeof(float) * 2);

    const ge::Shape softmaxShape({normalizeRow, normalizeColumn});
    const uint32_t minLocalWorkSize = AscendC::GetSoftMaxMinTmpSize(softmaxShape, sizeof(float), false);
    if (minLocalWorkSize > maxLocalWorkSize) {
        TEST_LOG("[ERROR] LocalWorkSize insufficient");
        return ge::GRAPH_FAILED; 
    }

    int normalizeLoop = qShape.GetDim(1) / normalizeRow;
    normalizeLoop = ((qShape.GetDim(1) % normalizeRow) == 0) ? normalizeLoop : normalizeLoop + 1;

    float dimSqrt = 1 / sqrt(qShape.GetDim(2));
    TEST_LOG("ub: %u; normalizeRow: %d; normalizeColumn: %d; dimSqrt: %f", ub, normalizeRow, normalizeColumn, dimSqrt);
    TEST_LOG("maxLocalWorkSize: %u; minLocalWorkSize: %u; normalizeLoop: %d",
                                                                    maxLocalWorkSize, minLocalWorkSize, normalizeLoop);

    // kvBmm Matmul Tilling
    matmul_tiling::MultiCoreMatmulTiling kvMatmulTiling(ascnedPlatform);
    kvMatmulTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    kvMatmulTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    kvMatmulTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    kvMatmulTiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    kvMatmulTiling.SetShape(qShape.GetDim(1), vShape.GetDim(2), kShape.GetDim(1));
    kvMatmulTiling.SetSingleShape(qShape.GetDim(1), vShape.GetDim(2), kShape.GetDim(1));
    std::cout << "kvMatmulTiling shape " << " " << qShape.GetDim(1) << " " << vShape.GetDim(2) << " " << kShape.GetDim(1) << std::endl;;
    kvMatmulTiling.SetBias(false);
    kvMatmulTiling.SetBufferSpace(-1, -1, -1);
    kvMatmulTiling.SetDim(coreNum);

    // set attr
    AttentionFusionTilingData tilingData;
    tilingData.set_normalizeAttr(attr);
    tilingData.set_attnDim(qShape.GetDim(2));
    tilingData.set_queryDim1(qShape.GetDim(1));
    tilingData.set_queryDim2(qShape.GetDim(2));
    tilingData.set_keyDim1(kShape.GetDim(1));
    tilingData.set_keyDim2(kShape.GetDim(2));
    tilingData.set_valueDim1(vShape.GetDim(1));
    tilingData.set_valueDim2(vShape.GetDim(2));
    tilingData.set_batchNum(qShape.GetDim(0));
    tilingData.set_normalizeLoop(normalizeLoop);
    tilingData.set_normalizeRow(normalizeRow);
    tilingData.set_normalizeColumn(normalizeColumn);
    tilingData.set_normalizeSqrt(dimSqrt);
    tilingData.set_maxSharedTmpBuf(maxLocalWorkSize);

    std::vector<int64_t> shapeVec = {16, 50 * 16};
    ge::Shape srcShape(shapeVec);
    AscendC::GetConfusionTransposeTilingInfo(srcShape, 0, sizeof(float), 7, tilingData.confusionTransposeTilingData);

    std::vector<int64_t> shapeVec1 = {56 * 16, 16};
    ge::Shape srcShape1(shapeVec1);
    AscendC::GetConfusionTransposeTilingInfo(srcShape1, 0, sizeof(float), 7, tilingData.confusionTransposeTilingData1);

    std::vector<int64_t> shapeVec2 = {16, 56 * 16};
    ge::Shape srcShape2(shapeVec2);
    AscendC::GetConfusionTransposeTilingInfo(srcShape2, 0, sizeof(float), 7, tilingData.confusionTransposeTilingData2);

    std::vector<int64_t> shapeVec3 = {50* 16, 16};
    ge::Shape srcShape3(shapeVec3);
    AscendC::GetConfusionTransposeTilingInfo(srcShape3, 0, sizeof(float), 7, tilingData.confusionTransposeTilingData3);

    // Get tilingData using on the kernel side 
    if (qkMatmulTiling.GetTiling(tilingData.qkMatmulTiling)) {
        return ge::GRAPH_FAILED;
    }
    if (kvMatmulTiling.GetTiling(tilingData.kvMatmulTiling)) {
        return ge::GRAPH_FAILED;
    }
    AscendC::SoftMaxTilingFunc(softmaxShape, sizeof(float), maxLocalWorkSize, tilingData.softMaxTilingData);

    context->SetBlockDim(coreNum);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* qShape = context->GetInputShape(0);
    const gert::Shape* kShape = context->GetInputShape(1);
    const gert::Shape* vShape = context->GetInputShape(2);

    gert::Shape* attnScoreShape = context->GetOutputShape(0);
    gert::Shape* softmaxOutShape = context->GetOutputShape(1);

    attnScoreShape->SetDimNum(3);
    attnScoreShape->SetDim(0, qShape->GetDim(0));
    attnScoreShape->SetDim(1, qShape->GetDim(1));
    attnScoreShape->SetDim(2, vShape->GetDim(2));

    softmaxOutShape->SetDimNum(3);
    softmaxOutShape->SetDim(0, qShape->GetDim(0));
    softmaxOutShape->SetDim(1, qShape->GetDim(1));
    softmaxOutShape->SetDim(2, kShape->GetDim(1));

    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDtype(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(1));
    return GRAPH_SUCCESS;
}
}


namespace ops {
class AttentionFusion : public OpDef {
public:
    explicit AttentionFusion(const char* name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("attn_mask")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("atten_score")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("softmax_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("mask_on").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDtype);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910");
    }
};

OP_ADD(AttentionFusion);
}
