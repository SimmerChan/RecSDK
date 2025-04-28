/* Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/

#include <algorithm>

#include "bounds_check_indices_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
constexpr int RESERVER_UB_SIZE = 20 * 1024;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const auto tableNumShape = context->GetInputShape(0)->GetStorageShape();
    const auto indicesShape = context->GetInputShape(1)->GetStorageShape();
    const auto offsetsShape = context->GetInputShape(2)->GetStorageShape();
    if (tableNumShape.GetDimNum() != 1 || indicesShape.GetDimNum() != 1 || offsetsShape.GetDimNum() != 1) {
        return ge::FAILED;
    }
    const int64_t numTable = tableNumShape.GetShapeSize();
    const int64_t totalNumBatch = offsetsShape.GetShapeSize() - 1;
    if (numTable == 0) {
        return ge::FAILED;
    }
    const int64_t numBatch = totalNumBatch / numTable;
    const int64_t numIndices = indicesShape.GetShapeSize();
    if (offsetsShape.GetShapeSize() != numBatch * numTable + 1) {
        return ge::FAILED;
    }
    const auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int64_t coreNum = ascendPlatform.GetCoreNumAiv();
    const int64_t coresPerTable = std::min(coreNum / numTable, numBatch);
    const int64_t blockDim = coresPerTable * numTable;
    if (blockDim <= 0) {
        return ge::FAILED;
    }
    uint64_t ubCanUsed;
    ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubCanUsed);
    ubCanUsed = ubCanUsed - RESERVER_UB_SIZE;

    BoundsCheckIndicesTilingData tiling;
    tiling.set_numTable(numTable);
    tiling.set_numIndices(numIndices);
    tiling.set_totalNumBatch(totalNumBatch);
    tiling.set_numBatch(numBatch);
    tiling.set_coresPerTable(coresPerTable);
    tiling.set_boundsCheckMode(*context->GetAttrs()->GetInt(0));
    tiling.set_ubSize(ubCanUsed);
    context->SetBlockDim(blockDim);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    gert::Shape* warningShape = context->GetOutputShape(0);
    gert::Shape shape = gert::Shape({100});
    *warningShape = shape;
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class BoundsCheckIndices : public OpDef {
public:
    explicit BoundsCheckIndices(const char* name) : OpDef(name)
    {
        this->Input("rows_per_table")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("offsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("warning")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("bounds_check_mode_int").Int();
        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(BoundsCheckIndices);
}  // namespace ops
