#include <cstdint>
#include <cstdio>
#include <iostream>
#include "tiling/platform/platform_ascendc.h"
#include "jagged_to_padded_dense_tiling.h"
#include "register/op_def_registry.h"

constexpr int GM_ALIGN = 64;
constexpr int RESERVER_UB_SIZE = 20 * 1024;
constexpr int DATA_TYPE_INT64=8;
constexpr int DATA_TYPE_INT32=4;
constexpr int DATA_TYPE_FLOAT32=4;
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto ascnedPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto valuesShape = context->GetInputShape(0)->GetStorageShape();
    auto offsetsShape = context->GetInputShape(1)->GetStorageShape();

    uint64_t ubCanUsed;
    ascnedPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubCanUsed);
    ubCanUsed = ubCanUsed - RESERVER_UB_SIZE;
    ubCanUsed = ubCanUsed/32/4*32*4;

    if (valuesShape.GetDimNum() != 2 or offsetsShape.GetDimNum()!=1) {
        printf("jagged_to_padded_dense_tiling is only used for values whit rank-3 and offset rank-1");
        return ge::FAILED;
    }

    size_t coreNum = ascnedPlatform.GetCoreNumAiv(); 

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    size_t systemWorkspacesSize = ascnedPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = systemWorkspacesSize;
    // 进行tiling, 为了保持尽量均匀，tailIndex之前长度为baseLen+1， 之后为baseLen
    // 例如7个数据分3个核，[3, 2, 2]
    int64_t totalBatch = offsetsShape.GetDim(0) - 1;
    int64_t baseBatchLen = (offsetsShape.GetDim(0) - 1)/coreNum;
    int64_t tailSplitIndex = (offsetsShape.GetDim(0) - 1)%coreNum;
    int64_t valuesDim0 = valuesShape.GetDim(0);
    int64_t valuesDim1 = valuesShape.GetDim(1);
    int64_t offsetDim0 = offsetsShape.GetDim(0);
    int64_t outDim1 = *context->GetAttrs()->GetInt(0);
    
    int64_t bytesOfDataType = 0;
    ge::DataType dataType = context->GetInputTensor(0)->GetDataType();
    if (dataType == ge::DataType::DT_FLOAT) {
        bytesOfDataType = DATA_TYPE_FLOAT32;
    } else {
        bytesOfDataType = DATA_TYPE_INT64;
    }

    int64_t offsetDataType = 4;
    ge::DataType offsetDataTypeGe = context->GetInputTensor(1)->GetDataType();
    if (offsetDataTypeGe == ge::DataType::DT_INT64) {
        offsetDataType = DATA_TYPE_INT64;
    } else {
        offsetDataType = DATA_TYPE_INT32;
    }

    JaggedToPaddedDenseTilingData tiling;
    tiling.set_totalBatch(totalBatch);
    tiling.set_baseBatchLen(baseBatchLen);
    tiling.set_tailSplitIndex(tailSplitIndex);
    tiling.set_valuesDim0(valuesDim0);
    tiling.set_valuesDim1(valuesDim1);
    tiling.set_offsetDim0(offsetDim0);
    tiling.set_outDim1(outDim1);
    tiling.set_ubCanUsed(ubCanUsed);
    tiling.set_bytesOfDataType(bytesOfDataType);
    tiling.set_offsetDataType(offsetDataType);

    // printf("totalBatch %ld  baseBatchLen %ld  tailSplitIndex %ld  valuesDim0 %ld  valuesDim1 %ld  offsetDim0 %ld ubCanUsed %ld  bytesOfDataType %ld  ",
    //         totalBatch, baseBatchLen, tailSplitIndex, valuesDim0, valuesDim1, offsetDim0, ubCanUsed, bytesOfDataType);

    context->SetBlockDim(coreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    int64_t maxLen = *context->GetAttrs()->GetInt(0);
    const gert::Shape* valuesShape = context->GetInputShape(0);
    const gert::Shape* offsetsShape = context->GetInputShape(1);

    gert::Shape* outShape = context->GetOutputShape(0);

    outShape->SetDimNum(3);
    outShape->SetDim(0, offsetsShape->GetDim(0));
    outShape->SetDim(1, maxLen);
    outShape->SetDim(2, valuesShape->GetDim(1));

    return GRAPH_SUCCESS;
}
}


namespace ops {
class JaggedToPaddedDense : public OpDef {
public:
    explicit JaggedToPaddedDense(const char* name) : OpDef(name)
    {
        this->Input("values")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("offsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("max_length").Int();
        this->Attr("padding_value").Float();

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910");
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(JaggedToPaddedDense);
}
