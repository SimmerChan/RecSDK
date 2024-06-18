#include <cstdint>
#include <cmath>
#include "attention_fusion_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#define TEST_LOG(fmt, args...) fprintf(stdout, fmt "\n", ##args)

namespace optiling {
#define ALIGN_32 (32 / sizeof(float))
#define RESERVER_UB_SIZE (20 * 1024)
#define ALREADY_ALIGNED 1
#define SPECIAL_ALIGNED_50 2
#define SPECIAL_K_DIM1 50
#define UB_TILES 3
#define DIM0 0
#define DIM1 1
#define DIM2 2

static int32_t MatmulTiling(gert::TilingContext* context, AttentionFusionTilingData &tilingData)
{
    // q (B, M, K) k (B, N, K) v (B, V, K)
    auto qShape = context->GetInputShape(0)->GetStorageShape();
    auto kShape = context->GetInputShape(1)->GetStorageShape();
    auto vShape = context->GetInputShape(2)->GetStorageShape();

    auto ascnedPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    // qkMatmul configuration
    matmul_tiling::MultiCoreMatmulTiling qkMm(ascnedPlatform);
    qkMm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    qkMm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    qkMm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    qkMm.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    qkMm.SetShape(qShape.GetDim(DIM1), kShape.GetDim(DIM1), kShape.GetDim(DIM2));

    qkMm.SetSingleShape(qShape.GetDim(DIM1), kShape.GetDim(DIM1), kShape.GetDim(DIM2));

    qkMm.SetBias(false);
    qkMm.SetBufferSpace(-1, -1, -1);
    qkMm.SetDim(coreNum);

    // kvBmm Matmul Tilling
    matmul_tiling::MultiCoreMatmulTiling kvMm(ascnedPlatform);
    kvMm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    kvMm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    kvMm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    kvMm.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    kvMm.SetShape(qShape.GetDim(DIM1), vShape.GetDim(DIM2), kShape.GetDim(DIM1));
    kvMm.SetSingleShape(qShape.GetDim(DIM1), vShape.GetDim(DIM2), kShape.GetDim(DIM1));

    kvMm.SetBias(false);
    kvMm.SetBufferSpace(-1, -1, -1);
    kvMm.SetDim(coreNum);

    // Get tilingData using on the kernel side 
    if (qkMm.GetTiling(tilingData.qkMatmulTiling) == -1 ||
        kvMm.GetTiling(tilingData.kvMatmulTiling) == -1) {
        return 1;
    }
    return 0;
}

static int32_t SoftmaxTiling(gert::TilingContext* context, AttentionFusionTilingData &tilingData, uint64_t ub)
{
    // q (B, M, K) k (B, N, K) v (B, V, K)
    auto qShape = context->GetInputShape(0)->GetStorageShape();
    auto kShape = context->GetInputShape(1)->GetStorageShape();
    auto vShape = context->GetInputShape(2)->GetStorageShape();

    int numOfelement = kShape.GetDim(DIM1) / ALIGN_32;
    uint8_t attr = 0;
    int normalizeColumn = 0;
    /* Get column with 32bytes alignment */
    if ((kShape.GetDim(DIM1) % ALIGN_32) == 0) {
        normalizeColumn = numOfelement * ALIGN_32;
        attr = ALREADY_ALIGNED;
    } else if (kShape.GetDim(DIM1) == SPECIAL_K_DIM1 && qShape.GetDim(DIM1) > 500) {
        normalizeColumn = numOfelement * ALIGN_32 + ALIGN_32;
        attr = SPECIAL_ALIGNED_50;
    } else {
        normalizeColumn = numOfelement * ALIGN_32 + ALIGN_32;
    }

    if ((sizeof(float) * normalizeColumn) > (ub / UB_TILES)) {
        TEST_LOG("[ERROR] key dim1 too large, LocalWorkSize insufficient");
        return 1;
    }

    /* Get how many rows that half of ub contains */
    int normalizeRow = ub / UB_TILES / (sizeof(float) * normalizeColumn);
    if (normalizeRow > qShape.GetDim(DIM1)) {
        normalizeRow = qShape.GetDim(DIM1);
    }

    if (attr == SPECIAL_ALIGNED_50) {
        normalizeRow = 16 * 16;
    }

    /* Get max Ub left for softmax shared tmp buffer */
    uint64_t maxLocalWorkSize = ub - (normalizeRow * normalizeColumn * sizeof(float) * 2);

    const ge::Shape softmaxShape({normalizeRow, normalizeColumn});
    const uint32_t minLocalWorkSize = AscendC::GetSoftMaxMinTmpSize(softmaxShape, sizeof(float), false);
    if (minLocalWorkSize > maxLocalWorkSize) {
        TEST_LOG("[ERROR] LocalWorkSize insufficient");
        return 1;
    }

    // divisor should not be 0
    if (normalizeRow == 0) {
        return 1;
    }
    int normalizeLoop = qShape.GetDim(DIM1) / normalizeRow;
    normalizeLoop = ((qShape.GetDim(DIM1) % normalizeRow) == 0) ? normalizeLoop : normalizeLoop + 1;
    float res = sqrt(qShape.GetDim(DIM2));
    float dimSqrt = (res != 0) ? (1 / res) : 0;

    // set attr
    tilingData.set_normalizeAttr(attr);
    tilingData.set_attnDim(qShape.GetDim(DIM2));
    tilingData.set_queryDim1(qShape.GetDim(DIM1));
    tilingData.set_queryDim2(qShape.GetDim(DIM2));
    tilingData.set_keyDim1(kShape.GetDim(DIM1));
    tilingData.set_keyDim2(kShape.GetDim(DIM2));
    tilingData.set_valueDim1(vShape.GetDim(DIM1));
    tilingData.set_valueDim2(vShape.GetDim(DIM2));
    tilingData.set_batchNum(qShape.GetDim(DIM0));
    tilingData.set_normalizeLoop(normalizeLoop);
    tilingData.set_normalizeRow(normalizeRow);
    tilingData.set_normalizeColumn(normalizeColumn);
    tilingData.set_normalizeSqrt(dimSqrt);
    tilingData.set_maxSharedTmpBuf(maxLocalWorkSize);

    AscendC::SoftMaxTilingFunc(softmaxShape, sizeof(float), maxLocalWorkSize, tilingData.softMaxTilingData);
    return 0;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    std::cout << "testing start " << std::endl;
    AttentionFusionTilingData tilingData;
    // Platform configuration
    auto ascnedPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    size_t systemWorkspacesSize = ascnedPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = 0 + systemWorkspacesSize;
    size_t coreNum = ascnedPlatform.GetCoreNumAic();

    uint64_t ub;
    ascnedPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    ub = ub - RESERVER_UB_SIZE;

    if (MatmulTiling(context, tilingData) != 0 || SoftmaxTiling(context, tilingData, ub) != 0) {
        return ge::GRAPH_FAILED;
    }

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
    attnScoreShape->SetDim(0, qShape->GetDim(DIM0));
    attnScoreShape->SetDim(1, qShape->GetDim(DIM1));
    attnScoreShape->SetDim(2, vShape->GetDim(DIM2));

    softmaxOutShape->SetDimNum(3);
    softmaxOutShape->SetDim(0, qShape->GetDim(DIM0));
    softmaxOutShape->SetDim(1, qShape->GetDim(DIM1));
    softmaxOutShape->SetDim(2, kShape->GetDim(DIM1));

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
