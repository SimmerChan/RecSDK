/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "register/op_def_registry.h"
#include "segment_sum_y_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static constexpr int ZDIM = 2;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SegmentSumYTilingData tiling;
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    const gert::StorageShape* x2_shape = context->GetInputShape(1);
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const auto coreNum = ascendPlatform.GetCoreNumAiv();
    tiling.set_batchSize(x1_shape->GetStorageShape().GetDim(0));
    tiling.set_yDim(x1_shape->GetStorageShape().GetDim(1));
    tiling.set_zDim(x1_shape->GetStorageShape().GetDim(ZDIM));
    tiling.set_segNum(x2_shape->GetStorageShape().GetDim(0) - 1);
    tiling.set_dimValue(x1_shape->GetStorageShape().GetDimNum());
    context->SetBlockDim(coreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    const gert::Shape* x2_shape = context->GetInputShape(1);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = {x1_shape->GetDim(0), x2_shape->GetDim(0) - 1, x1_shape->GetDim(2)};
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class SegmentSumY : public OpDef {
public:
    explicit SegmentSumY(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("segIndices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(SegmentSumY);
}  // namespace ops
