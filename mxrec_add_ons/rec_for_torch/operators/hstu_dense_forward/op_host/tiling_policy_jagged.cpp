/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

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
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>
#include <cassert>

#include "register/op_def_registry.h"
#include "tiling_policy_factory.h"
#include "tiling_policy_jagged.h"

constexpr bool JAGGED_TASK_ASSIGN_DEBUG = false;

#if JAGGED_TASK_ASSIGN_DEBUG
#include <chrono>
#endif

constexpr uint32_t CONST_2 = 2;

namespace HstuDenseForward {

REGISTER_POLICY(LAYOUT_TYPE::JAGGED, std::make_shared<TilingPolicyJagged>());

bool TilingPolicyJagged::TilingShape(gert::TilingContext* context, optiling::HstuDenseForwardTilingData &tiling)
{
    int64_t batchSize;
    int64_t headNum;
    int64_t headDIM;
    int64_t seqLens;

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_CHECK_PTR_NULL(attrs, return false);

    const auto seqOffset = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_T::INDEX_4);
    OPS_CHECK_PTR_NULL(seqOffset, return false);

    auto *seqOffsetData = const_cast<int64_t *>(reinterpret_cast<const int64_t *>(seqOffset->GetData()));
    OPS_CHECK_PTR_NULL(seqOffsetData, return false);

    int64_t seqOffsetLens = seqOffset->GetSize();
    batchSize = seqOffsetLens - 1;
    OPS_CHECK(batchSize > MAX_BATCH_SIZE,
        OPS_LOG_E("", "batch size is over limit %d", MAX_BATCH_SIZE), return false);

    auto queryShape = context->GetInputShape(INDEX_T::INDEX_0)->GetStorageShape();
    headNum = queryShape.GetDim(INDEX_T::INDEX_1);
    headDIM = queryShape.GetDim(INDEX_T::INDEX_2);
    seqLens = tiling.get_maxSeqLen();

    tiling.set_batchSize(batchSize);
    tiling.set_headNum(headNum);
    tiling.set_dim(headDIM);
    tiling.set_seqLen(seqLens);

    OPS_CHECK(!GeneralShapeCheck(batchSize, seqLens, headNum, headDIM),
        OPS_LOG_E("", "Jagged Shape Check failed"), return false);
    return true;
}

// static void CallBlockAssign(
//     uint32_t *seqOffsets,
//     uint32_t coreNum,
//     std::vector<BlockTaskInfo> &workTasks,
//     std::vector<int> &workLoads,
//     optiling::HstuDenseForwardTilingData &tiling)
// {
//     uint32_t batchSize = tiling.get_batchSize();
//     uint32_t headNum = tiling.get_headNum();
//     uint32_t maskType = tiling.get_maskType();

//     auto taskAssigner = BlockTaskAssign(seqOffsets, coreNum, BLOCK_HEIGHT, batchSize, headNum);
//     if (maskType == 0) {
//         taskAssigner.ComputeCausal(workTasks, workLoads);
//     } else {
//         taskAssigner.Compute(workTasks, workLoads);
//     }
// }

bool TilingPolicyJagged::TilingCore(gert::TilingContext* context, optiling::HstuDenseForwardTilingData &tiling)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_CHECK_PTR_NULL(attrs, return false);

    const auto seqOffset = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_T::INDEX_4);
    OPS_CHECK_PTR_NULL(seqOffset, return false);

    auto *seqOffsetData = const_cast<int64_t *>(reinterpret_cast<const int64_t *>(seqOffset->GetData()));
    int seqOffsetLens = seqOffset->GetSize();
    if (seqOffsetLens > (MAX_BATCH_SIZE + 1)) {
        OPS_LOG_E("", "seqOffsetLens exceed limit %d \n", MAX_BATCH_SIZE + 1);
        return false;
    }

// #if JAGGED_TASK_ASSIGN_DEBUG
//     auto start = std::chrono::high_resolution_clock::now();
// #endif

    // std::vector<BlockTaskInfo> workTasks;
    // std::vector<int> workLoads;

    // uint32_t seqOffsets[MAX_BATCH_SIZE + 1] = {0};
    // for (auto i = 0; i < seqOffsetLens; i++) {
    //     seqOffsets[i] = seqOffsetData[i];
    // }

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAiv();

    // CallBlockAssign(seqOffsets, coreNum, workTasks, workLoads, tiling);

// #if JAGGED_TASK_ASSIGN_DEBUG
//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double, std::micro> elapsed = end - start;
//     std::cout << "BlockTaskAssign Elapsed time: " << elapsed.count() << " us\n";

//     for (auto i = 0; i < coreNum; i++) {
//         OPS_LOG_E("", "aicore :%d startBlockId:%d endBlockId:%d totalTaskNumber:%d\n",
//             i, workTasks[i].startBlockId, workTasks[i].endBlockId, workLoads[i]);
//     }
// #endif

    // uint32_t startBlockId[MAX_AIV_NUM] = {0};
    // uint32_t endBlockId[MAX_AIV_NUM] = {0};

    // for (auto i = 0; i < coreNum; i++) {
    //     startBlockId[i] = workTasks[i].startBlockId;
    //     endBlockId[i] = workTasks[i].endBlockId;
    // }

    tiling.set_seqOffset(seqOffsets);
    // tiling.set_eachCoreStartBlockId(startBlockId);
    // tiling.set_eachCoreEndBlockId(endBlockId);

    size_t aicCoreNum = ascendPlatform.GetCoreNumAic();
    context->SetBlockDim(aicCoreNum);
    
    return true;
}

bool TilingPolicyJagged::TilingKeySet(gert::TilingContext* context, optiling::HstuDenseForwardTilingData &tiling)
{
    ge::DataType qTypeGe = context->GetInputTensor(0)->GetDataType();
    if (qTypeGe == ge::DataType::DT_FLOAT) {
        context->SetTilingKey(JAGGED_FLOAT_TILING_KEY);
    } else if (qTypeGe == ge::DataType::DT_FLOAT16) {
        context->SetTilingKey(JAGGED_FLOAT16_TILING_KEY);
    } else if (qTypeGe == ge::DataType::DT_BF16) {
        context->SetTilingKey(JAGGED_BF16_TILING_KEY);
    } else {
        OPS_LOG_E("", "invalid datatype, only support fp32, fp16, bf16");
        return false;
    }

    return true;
}

void TilingPolicyJagged::DumpTiling(optiling::HstuDenseForwardTilingData &tiling)
{
    this->TilingPolicy::DumpTiling(tiling);

    // uint32_t *seqOffset = tiling.get_seqOffset();
    // uint32_t *startBlockId = tiling.get_eachCoreStartBlockId();
    // uint32_t *endBlockId = tiling.get_eachCoreEndBlockId();

    // OPS_LOG_D("seq offset:");
    // for (auto i = 0; i < (tiling.get_batchSize() + 1); i++) {
    //     OPS_LOG_D("%d ", seqOffset[i]);
    // }
    // OPS_LOG_D("\n");

    // OPS_LOG_D("core block range:\n");
    // for (auto i = 0; i < MAX_AIV_NUM; i++) {
    //     OPS_LOG_E("", "core_id:%d startBlockId:%d endBlockId:%d\n", i, startBlockId[i], endBlockId[i]);
    // }
}

}