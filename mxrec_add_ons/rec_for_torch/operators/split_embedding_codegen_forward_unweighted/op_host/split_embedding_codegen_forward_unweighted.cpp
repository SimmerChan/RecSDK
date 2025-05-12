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

#include <cstdint>
#include <cstdio>

#include "register/op_def_registry.h"
#include "split_embedding_codegen_forward_unweighted_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

constexpr int DATA_TYPE_FLOAT32 = 0;
constexpr int DATA_TYPE_INT64 = 1;

constexpr int RESERVER_UB_SIZE = 20 * 1024;
constexpr int UB_ALIGN = 32;
constexpr int NUM_QUEUE = 32;
constexpr int POOL_MODE_NOBAG = 2;

constexpr int DEV_WEIGHTS_INDEX = 0;
constexpr int UVM_WEIGHTS_INDEX = 1;
constexpr int LXU_CACHE_WEIGHTS_INDEX = 2;
constexpr int WEIGHTS_PLACEMENTS_INDEX = 3;
constexpr int WEIGHTS_OFFSETS_INDEX = 4;
constexpr int D_OFFSETS_INDEX = 5;
constexpr int INDICES_INDEX = 6;
constexpr int OFFSETS_INDEX = 7;
constexpr int LXU_CACHE_LOCATIONS_INDEX = 8;
constexpr int HASH_INDICES_INDEX = 9;
constexpr int POOL_MODE_INDEX = 2;
constexpr int MAX_D_INDEX = 1;
constexpr int EC_KEY = 1;
constexpr int EBC_KEY = 2;

static ge::graphStatus ShapeTilingFunc(gert::TilingContext* context,
                                       SplitEmbeddingCodegenForwardUnweightedTilingData& tilingData)
{
    int64_t devWeightsDim0 = context->GetInputShape(DEV_WEIGHTS_INDEX)->GetStorageShape().GetDim(0);
    int64_t weightsOffsetsDim0 = context->GetInputShape(WEIGHTS_OFFSETS_INDEX)->GetStorageShape().GetDim(0);
    int64_t dOffsetsDim0 = context->GetInputShape(D_OFFSETS_INDEX)->GetStorageShape().GetDim(0);
    int64_t indicesDim0 = context->GetInputShape(INDICES_INDEX)->GetStorageShape().GetDim(0);
    int64_t offsetsDim0 = context->GetInputShape(OFFSETS_INDEX)->GetStorageShape().GetDim(0);
    int64_t poolMode = *context->GetAttrs()->GetInt(POOL_MODE_INDEX);

    if (weightsOffsetsDim0 == 0) {
        printf("weightsOffsetsDim0 is 0;");
        return ge::FAILED;
    }
    if (dOffsetsDim0 <= 1) {
        printf("dOffsetsDim0 is error;");
        return ge::FAILED;
    }

    auto hashIndices = context->GetOptionalInputTensor(HASH_INDICES_INDEX);
    if (hashIndices == nullptr) {
        tilingData.set_enableHash(0);
    } else {
        tilingData.set_enableHash(1);
        indicesDim0 = context->GetInputShape(HASH_INDICES_INDEX)->GetStorageShape().GetDim(0);
    }

    // bag
    int64_t outDim0 = (offsetsDim0 - 1) / weightsOffsetsDim0;
    int64_t outDim1 = *context->GetAttrs()->GetInt(0);
    if (poolMode == POOL_MODE_NOBAG) { // no bag
        outDim0 = indicesDim0;
        outDim1 = *context->GetAttrs()->GetInt(MAX_D_INDEX);
    }

    int64_t bytesOfDataType = sizeof(float);
    int64_t offsetDataType = DATA_TYPE_INT64;

    tilingData.set_devWeightsDim0(devWeightsDim0);
    tilingData.set_weightsOffsetsDim0(weightsOffsetsDim0);
    tilingData.set_dOffsetsDim0(dOffsetsDim0);
    tilingData.set_indicesDim0(indicesDim0);
    tilingData.set_offsetsDim0(offsetsDim0);
    tilingData.set_outDim0(outDim0);
    tilingData.set_outDim1(outDim1);
    tilingData.set_bytesOfDataType(bytesOfDataType);
    tilingData.set_offsetDataType(offsetDataType);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto ascnedPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    size_t systemWorkspacesSize = ascnedPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = systemWorkspacesSize;

    SplitEmbeddingCodegenForwardUnweightedTilingData tiling;

    // Shape and dType
    if (ShapeTilingFunc(context, tiling) != ge::GRAPH_SUCCESS) {
        return ge::FAILED;
    }

    // Tiling
    size_t coreNum = ascnedPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        printf("Core num is 0;");
        return ge::FAILED;
    }

    int64_t splitBaseLen = (tiling.get_offsetsDim0() - 1) / coreNum;
    int64_t tailSplitIndex = (tiling.get_offsetsDim0() - 1)% coreNum;

    tiling.set_splitBaseLen(splitBaseLen);
    tiling.set_tailSplitIndex(tailSplitIndex);

    uint64_t ubCanUsed;
    ascnedPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubCanUsed);
    ubCanUsed = ubCanUsed - RESERVER_UB_SIZE;
    tiling.set_ubCanUsed(ubCanUsed);

    int64_t poolMode = *context->GetAttrs()->GetInt(POOL_MODE_INDEX);
    int64_t maxD = *context->GetAttrs()->GetInt(MAX_D_INDEX);

    tiling.set_poolMode(poolMode);
    tiling.set_maxD(maxD);

    context->SetBlockDim(coreNum);
    context->SetNeedAtomic(true);
    if (context->GetRawTilingData() == nullptr) {
        printf("GetRawTilingData Failed!");
        return ge::FAILED
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class SplitEmbeddingCodegenForwardUnweighted : public OpDef {
public:
    explicit SplitEmbeddingCodegenForwardUnweighted(const char* name) : OpDef(name)
    {
        this->Input("dev_weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("uvm_weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("lxu_cache_weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weights_placements")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weights_offsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("D_offsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
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
        this->Input("lxu_cache_locations")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("hash_indices")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("total_D").Int();
        this->Attr("max_D").Int();
        this->Attr("pool_mode").Int();
        this->Attr("output_dtype").Int();
        this->Attr("is_experimental").Int();

        this->SetInferShape(ge::InferShape);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910");
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(SplitEmbeddingCodegenForwardUnweighted);
}  // namespace ops
