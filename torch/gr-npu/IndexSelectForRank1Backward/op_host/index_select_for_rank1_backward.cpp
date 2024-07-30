/**
 * @file index_select_for_rank1_backward.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "tiling/platform/platform_ascendc.h"
#include "index_select_for_rank1_backward_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

constexpr int GM_ALIGN = 64;
constexpr int FLOAT_BYTESIZE = 4;
    
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto gardShape = context->GetInputShape(0)->GetStorageShape();
    auto xShape = context->GetInputShape(1)->GetStorageShape();
    auto indexShape = context->GetInputShape(2)->GetStorageShape();

    if (xShape.GetDimNum() != 1) {
        printf("IndexSectForRank1Backward is only used for input-1 with dim 0 but is %ld", xShape.GetDimNum());
        return ge::FAILED;
    }

    int64_t keyDim0Align64B = (xShape.GetShapeSize()*FLOAT_BYTESIZE+GM_ALIGN-1)/GM_ALIGN*GM_ALIGN/FLOAT_BYTESIZE;
    auto ascnedPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascnedPlatform.GetCoreNumAiv();

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    size_t systemWorkspacesSize = ascnedPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = systemWorkspacesSize + coreNum*keyDim0Align64B*FLOAT_BYTESIZE;

    // 进行tiling, 为了保持尽量均匀，tailIndex之前长度为baseLen+1， 之后为baseLen
    // 例如7个数据分3个核，[3, 2, 2]
    if (coreNum == 0) {
        return ge::FAILED;
    }
    int64_t totalLen = indexShape.GetShapeSize();
    int64_t xDim0 = xShape.GetShapeSize();
    int64_t baseLen = indexShape.GetShapeSize() / coreNum;
    int64_t tailSplitIndex = indexShape.GetShapeSize() % coreNum;

    IndexSelectForRank1BackwardTilingData tiling;
    tiling.set_totalLen(totalLen);
    tiling.set_xDim0(xDim0);
    tiling.set_baseLen(baseLen);
    tiling.set_tailSplitIndex(tailSplitIndex);
    tiling.set_keyDim0Align64B(keyDim0Align64B);

    context->SetBlockDim(coreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
    static ge::graphStatus InferShape(gert::InferShapeContext* context)
    {
        const gert::Shape* xShape = context->GetInputShape(2);
        const gert::Shape* indexShape = context->GetInputShape(1);

        gert::Shape* gradXShape = context->GetOutputShape(0);
        gert::Shape* gradIndexShape = context->GetOutputShape(1);

        *gradXShape = *xShape;
        *gradIndexShape = *indexShape;
        return GRAPH_SUCCESS;
    }
}


namespace ops {
class IndexSelectForRank1Backward : public OpDef {
public:
    explicit IndexSelectForRank1Backward(const char* name) : OpDef(name)
    {
        this->Input("grad_y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("grad_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("grad_index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910");
    }
};

OP_ADD(IndexSelectForRank1Backward);
}
