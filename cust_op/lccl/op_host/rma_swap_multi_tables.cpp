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
#include "rma_swap_multi_tables_tiling.h"
#include "register/op_def_registry.h"
#include "rma_log.h"

constexpr int32_t BLOCK_DIM = 16; // 至少需要4个core

namespace optiling {
    constexpr int32_t RMA_DIM_MAX = 2;
    constexpr int32_t RMA_WORK_SPACE_SIZE = 202 * 1024 * 1024;

    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        LOG_DEBUG("Rma TilingFunc");
        RmaSwapMultiTablesTilingData tiling;
        context->SetBlockDim(BLOCK_DIM);
        context->SetTilingKey(1);

        auto dimNum = context->GetInputShape(0)->GetStorageShape().GetDimNum();

        // dimValue是一个attr，不能使用vector，只能用数组
        uint64_t dims[RMA_DIM_MAX] = {0};
        if (dimNum == 1) {
            dims[0] = 1;
            dims[1] = context->GetInputShape(0)->GetStorageShape().GetDim(0);
        } else if (dimNum == 2) {
            dims[0] = context->GetInputTensor(4)->GetShapeSize();
            dims[1] = context->GetInputShape(0)->GetStorageShape().GetDim(1);
        } else {
            LOG_ERROR("dim-num %d is invalid", dimNum);
            return ge::GRAPH_FAILED;
        }
        tiling.set_dimValue(dims);
        tiling.set_dimNum(RMA_DIM_MAX);

        uint64_t size = context->GetInputTensor(3)->GetShapeSize();
        tiling.set_updateLen(size);

        auto attrs = context->GetAttrs();
        if (attrs == nullptr) {
            return ge::GRAPH_FAILED;
        }

        auto gmem_attr_in = attrs->GetStr(0);
        int32_t *shm_swap_in = (int32_t *)(std::stoul(gmem_attr_in));
        if (shm_swap_in == nullptr) {
            return ge::GRAPH_FAILED;
        }
        tiling.set_shmSwapIn((uint64_t)shm_swap_in);

        auto gmem_attr_out = attrs->GetStr(1);
        int32_t *shm_swap_out = (int32_t *)(std::stoul(gmem_attr_out));
        if (shm_swap_out == nullptr) {
            return ge::GRAPH_FAILED;
        }
        tiling.set_shmSwapOut((uint64_t)shm_swap_out);

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
        const gert::Shape *x1_shape = context->GetInputShape(0);
        gert::Shape *y_shape = context->GetOutputShape(0);
        if (y_shape == nullptr) {
            LOG_ERROR("output shape is null");
            return ge::GRAPH_FAILED;
        }
        y_shape->SetDimNum(1);
        y_shape->SetDim(0, BLOCK_DIM);
//        y_shape->SetDimNum(2);
//        y_shape->SetDim(0, 300000);
//        y_shape->SetDim(0, 32);

        return GRAPH_SUCCESS;
    }
//static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
//{
//    LOG_DEBUG("RmaSwap InferDataType");
//    auto ret = context->SetOutputDataType(0, ge::DataType::DT_INT64);
//    return GRAPH_SUCCESS;
//}
}

namespace ops {
    class RmaSwapMultiTables : public OpDef {
    public:
        explicit RmaSwapMultiTables(const char* name) : OpDef(name)
        {
            this->Input("table0")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT })
                .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
            this->Input("table1")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT })
                .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
            this->Input("table2")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT })
                .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
            this->Input("swap_in_index")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_INT64, ge::DT_INT64, ge::DT_INT64 })
                .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
            this->Input("swap_out_index")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_INT64, ge::DT_INT64, ge::DT_INT64 })
                .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
            this->Output("output")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_INT64, ge::DT_INT64, ge::DT_INT64 })
                .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
//            this->Output("output")
//                .ParamType(REQUIRED)
//                .DataType({ ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT })
//                .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
//                .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
            this->Attr("shm_swap_in").String();
            this->Attr("shm_swap_out").String();

            this->SetInferShape(ge::InferShape);
//        this->SetInferDataType(ge::InferDataType);

            this->AICore().SetTiling(optiling::TilingFunc);
//            this->AICore().AddConfig("ascend910b");
            this->AICore().AddConfig("ascend910_93");
        }
    };

    OP_ADD(RmaSwapMultiTables);
}
