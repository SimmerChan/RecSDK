/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

#include "tiling/platform/platform_ascendc.h"
#include "rma_swap_tiling.h"
#include "register/op_def_registry.h"
#include "rma_log.h"

namespace optiling {
    constexpr int32_t RMA_DIM_MAX = 2;
    constexpr int32_t BLOCK_DIM = 4; // 至少需要4个core
    constexpr int32_t RMA_WORK_SPACE_SIZE = 22 * 1024 * 1024;

    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        LOG_DEBUG("Rma TilingFunc");
        RmaSwapTilingData tiling;
        context->SetBlockDim(BLOCK_DIM);
        context->SetTilingKey(1);

        auto dimNum = context->GetInputShape(0)->GetStorageShape().GetDimNum();

        // dimValue是一个attr，不能使用vector，只能用数组
        uint64_t dims[RMA_DIM_MAX] = {0};
        if (dimNum == 1) {
            dims[0] = 1;
            dims[1] = context->GetInputShape(0)->GetStorageShape().GetDim(0);
        } else if (dimNum == 2) {
            for (int32_t i = 0; i < dimNum; i++) {
                dims[i] = context->GetInputShape(0)->GetStorageShape().GetDim(i);
            }
        } else {
            LOG_ERROR("dim-num %d is invalid", dimNum);
            return ge::GRAPH_FAILED;
        }
        tiling.set_dimValue(dims);
        tiling.set_dimNum(RMA_DIM_MAX);

        uint64_t size = context->GetInputTensor(1)->GetShapeSize();
        tiling.set_updateLen(size);

        auto attrs = context->GetAttrs();
        if (attrs == nullptr) {
            return ge::GRAPH_FAILED;
        }

        auto gmemAttrIn = attrs->GetStr(0);
        int32_t *shmSwapIn = (int32_t *)(std::stoul(gmemAttrIn));
        if (shmSwapIn == nullptr) {
            return ge::GRAPH_FAILED;
        }
        tiling.set_shmSwapIn((uint64_t)shmSwapIn);

        auto gmemAttrOut = attrs->GetStr(1);
        int32_t *shmSwapOut = (int32_t *)(std::stoul(gmemAttrOut));
        if (shmSwapOut == nullptr) {
            return ge::GRAPH_FAILED;
        }
        tiling.set_shmSwapOut((uint64_t)shmSwapOut);

        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = RMA_WORK_SPACE_SIZE + sysWorkspaceSize;

        return ge::GRAPH_SUCCESS;
    }
}

namespace ge {
    static ge::graphStatus InferShape(gert::InferShapeContext *context)
    {
        LOG_DEBUG("RmaSwap InferShape");
        gert::Shape *outputShape = context->GetOutputShape(0);
        if (outputShape == nullptr) {
            LOG_ERROR("output shape is null");
            return ge::GRAPH_FAILED;
        }
        outputShape->SetDimNum(1);
        outputShape->SetDim(0, 8);

        return GRAPH_SUCCESS;
    }
}

namespace ops {
    class RmaSwap : public OpDef {
    public:
        explicit RmaSwap(const char* name) : OpDef(name)
        {
            this->Input("update_table")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT })
                .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
            this->Input("update_index")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_INT64, ge::DT_INT64, ge::DT_INT64 })
                .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
            this->Output("output")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_INT64, ge::DT_INT64, ge::DT_INT64 })
                .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
            this->Attr("shm_swap_in").String();
            this->Attr("shm_swap_out").String();

            this->SetInferShape(ge::InferShape);

            this->AICore().SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend910b");
            this->AICore().AddConfig("ascend910_93");
        }
    };

    OP_ADD(RmaSwap);
}
