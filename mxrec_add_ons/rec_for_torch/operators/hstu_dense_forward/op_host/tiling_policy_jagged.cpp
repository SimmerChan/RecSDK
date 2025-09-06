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

constexpr int QKV_DIM = 3;

namespace {
template<typename T>
inline T ceilDiv(T dividend, T divisor) {
    return (dividend + divisor - 1) / divisor;
}

struct MaskParams {
    uint32_t *seqOffsets;
    int64_t batchSize;
    int64_t headNum;
    int64_t numContext;
    int64_t numTarget;
    int64_t targetGroupSize;
    int64_t blockHeight;
};

struct BlockTaskInfo {
    uint32_t startBlockId = 0;
    uint32_t endBlockId = 0;

    friend std::ostream& operator<<(std::ostream& os, const BlockTaskInfo& blockTask)
    {
        return os << "startBlockId:" << blockTask.startBlockId << " " <<
               "endBlockId:" << blockTask.endBlockId << " ";
    }
};

class MaskPolicy {
public:
    virtual int64_t GetSeqTasks(int64_t seqlen, MaskParams& params) = 0;
    virtual uint32_t InitTaskNum(const std::vector<int64_t>& blockNumber, uint32_t batchId, MaskParams& params) = 0;
    virtual void UpdateTask(uint32_t& taskNum) = 0;
};

class FullMaskPolicy : public MaskPolicy {
public:
    int64_t GetSeqTasks(int64_t seqlen, MaskParams& params) override {
        auto blkNum = ceilDiv(seqlen, params.blockHeight);
        return blkNum * blkNum;
    }

    uint32_t InitTaskNum(const std::vector<int64_t>& blockNumber, uint32_t batchId, MaskParams& params) override {
        return blockNumber[batchId];
    }

    void UpdateTask(uint32_t& taskNum) override {
        // 无变化
    }
};

class CausalMaskPolicy : public MaskPolicy {
public:
    bool initFlag = false;
    int64_t GetSeqTasks(int64_t seqlen, MaskParams& params) override {
        // causal mask计算量
        int64_t blkNum = ceilDiv(seqlen, params.blockHeight);
        int64_t task = blkNum * (blkNum + 1) / 2;
        // context mask计算量
        if (params.numContext > 0) {
            int64_t contextWidth = ceilDiv(seqlen - params.numTarget, params.blockHeight) - 1;  // -1避免重复计算
            task += contextWidth;  // 仅考虑numContext < blockHeight的情况，contextHeight = 1
        }
        return task;
    }

    uint32_t InitTaskNum(const std::vector<int64_t>& blockNumber, uint32_t batchId, MaskParams& params) override {
        this->initFlag = true;
        uint32_t taskNums = blockNumber[batchId] - ceilDiv(params.numTarget, params.blockHeight);  // 任务量估算(误差0~1)
        taskNums = (taskNums > 0) ? taskNums : 1;
        return taskNums;
    }

    void UpdateTask(uint32_t& taskNum) override {
        if (this->initFlag) {
            this->initFlag = false;
            taskNum = 1;
        }
        taskNum++;
    }
};

template<typename Policy>
class BlockTaskAssign {
public:
    BlockTaskAssign(MaskParams& params, uint32_t coreNum)
    {
#if JAGGED_TASK_ASSIGN_DEBUG
        OPS_LOG_D("BlockTaskAssign coreNum:%d blockLen:%d batchSize:%d headNum:%d\n",
                  coreNum, params.blockHeight, params.batchSize, params.headNum);
        OPS_LOG_D("BlockTaskAssign seqOffsets:");
        for (auto i = 0; i <= batchSize; i++) {
            OPS_LOG_D("%d ", params.seqOffsets[i]);
        }
        OPS_LOG_D("\n");
#endif
        this->policy = Policy();
        this->params = params;
        this->seqOffsets = params.seqOffsets;
        this->coreNum = coreNum;
        this->blockLen = params.blockHeight;
        this->batchSize = params.batchSize;
        this->headNum = params.headNum;

        this->numContext = params.numContext;
        this->numTarget = params.numTarget;
        this->targetGroupSize = params.targetGroupSize;
    }

    void Compute(std::vector<BlockTaskInfo>& workTasks, std::vector<int>& workLoads) {
        uint32_t bXn = batchSize * headNum;
        std::vector<int64_t> blockNumber(bXn, 0);
        ComputeBlockNum(this->seqOffsets, blockNumber);

        uint32_t totalTaskNumber = 0;
        for (auto seqId = 0; seqId < batchSize; seqId++) {
            auto seqlen = seqOffsets[seqId + 1] - seqOffsets[seqId];
            totalTaskNumber += this->policy.GetSeqTasks(seqlen, this->params);
        }

        uint32_t eachCoreTaskNumLimit = ceilDiv(totalTaskNumber, coreNum);

#if JAGGED_TASK_ASSIGN_DEBUG
        int64_t total_block_number = 0;
        for (auto x : blockNumber) {
            total_block_number += x;
        }
        OPS_LOG_D("eachCoreTaskNumLimit :%d totalTaskNumber:%d total_block_number:%d\n",
                  eachCoreTaskNumLimit, totalTaskNumber, total_block_number);
#endif
        DistributeTasks(workTasks, workLoads, eachCoreTaskNumLimit, blockNumber);
    }

private:
    Policy policy;
    MaskParams params;
    uint32_t *seqOffsets = nullptr;
    uint32_t coreNum = 0;
    uint32_t blockLen = 0;
    uint32_t batchSize = 0;
    uint32_t headNum = 0;
    uint32_t numContext = 0;
    uint32_t numTarget = 0;
    uint32_t targetGroupSize = 0;

    void ComputeBlockNum(uint32_t* seqOffsets, std::vector<int64_t>& blockNumber)
    {
        for (auto seqId = 0; seqId < batchSize; seqId++) {
            auto batchBlockSize = seqOffsets[seqId + 1] - seqOffsets[seqId];
            int64_t blk = ceilDiv(batchBlockSize, blockLen);
            for (auto headId = 0; headId < headNum; headId++) {
                blockNumber[seqId * headNum + headId] = blk;
            }
        }
    }

    bool BatchSwitch(
        std::vector<int64_t> &blockNumber,
        uint32_t &batchId,
        uint32_t totalBatchSize,
        uint32_t &taskNum)
    {
        if (blockNumber[batchId] == 0) {
            batchId++;
            if (batchId >= totalBatchSize) {
                return false;  // 循环结束标志
            }
            taskNum = this->policy.InitTaskNum(blockNumber, batchId, params);
        }
        return true;  // 进入下一循环
    }

    void DistributeTasks(std::vector<BlockTaskInfo>& workTasks,
                         std::vector<int>& workLoads,
                         int64_t eachCoreTaskNumLimit,
                         std::vector<int64_t>& blockNumber
                         )
    {
        uint32_t bXn = batchSize * headNum;

        workTasks.resize(coreNum);
        workLoads.resize(coreNum, 0);

        uint32_t batchId = 0;
        uint32_t taskNum = this->policy.InitTaskNum(blockNumber, batchId, params);
        uint32_t processBlockNum = 0;
        uint32_t processTaskNum = 0;
        for (int i = 0; i < coreNum && batchId < bXn; i++) {
            BlockTaskInfo blockTask;
            blockTask.startBlockId = processBlockNum;

            while (workLoads[i] < eachCoreTaskNumLimit) {
                workLoads[i] += taskNum;
                processTaskNum += taskNum;
                processBlockNum++;
                blockNumber[batchId]--;

                this->policy.UpdateTask(taskNum);
                if (!BatchSwitch(blockNumber, batchId, bXn, taskNum)) {
                    break;
                }
            }

            blockTask.endBlockId = processBlockNum;
            workTasks[i] = blockTask;
        }

#if JAGGED_TASK_ASSIGN_DEBUG
        OPS_LOG_D("processTaskNum :%d processBlockNum:%d\n", processTaskNum, processBlockNum);
        assert(processTaskNum == totalTaskNumber);
        assert(processBlockNum == total_block_number);
#endif
    }
};
}

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

static void CallBlockAssign(
    uint32_t *seqOffsets,
    uint32_t coreNum,
    std::vector<BlockTaskInfo> &workTasks,
    std::vector<int> &workLoads,
    optiling::HstuDenseForwardTilingData &tiling)
{
    MaskParams params;
    params.seqOffsets = seqOffsets;
    params.batchSize = tiling.get_batchSize();
    params.headNum = tiling.get_headNum();
    params.numContext = tiling.get_numContext();
    params.numTarget = tiling.get_numTarget();
    params.targetGroupSize = tiling.get_targetGroupSize();
    params.blockHeight = BLOCK_HEIGHT;

    uint32_t maskType = tiling.get_maskType();
    if (maskType == 0) {
        auto taskAssigner = BlockTaskAssign<CausalMaskPolicy>(params, coreNum);
        taskAssigner.Compute(workTasks, workLoads);
    } else {
        auto taskAssigner = BlockTaskAssign<FullMaskPolicy>(params, coreNum);
        taskAssigner.Compute(workTasks, workLoads);
    }
}

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

#if JAGGED_TASK_ASSIGN_DEBUG
    auto start = std::chrono::high_resolution_clock::now();
#endif

    std::vector<BlockTaskInfo> workTasks;
    std::vector<int> workLoads;

    uint32_t seqOffsets[MAX_BATCH_SIZE + 1] = {0};
    for (auto i = 0; i < seqOffsetLens; i++) {
        seqOffsets[i] = seqOffsetData[i];
    }

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAiv();

    CallBlockAssign(seqOffsets, coreNum, workTasks, workLoads, tiling);

#if JAGGED_TASK_ASSIGN_DEBUG
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> elapsed = end - start;
    std::cout << "BlockTaskAssign Elapsed time: " << elapsed.count() << " us\n";

    for (auto i = 0; i < coreNum; i++) {
        OPS_LOG_E("", "aicore :%d startBlockId:%d endBlockId:%d totalTaskNumber:%d\n",
                  i, workTasks[i].startBlockId, workTasks[i].endBlockId, workLoads[i]);
    }
#endif

    uint32_t startBlockId[MAX_AIV_NUM] = {0};
    uint32_t endBlockId[MAX_AIV_NUM] = {0};

    for (auto i = 0; i < coreNum; i++) {
        startBlockId[i] = workTasks[i].startBlockId;
        endBlockId[i] = workTasks[i].endBlockId;
    }

    tiling.set_seqOffset(seqOffsets);
    tiling.set_eachCoreStartBlockId(startBlockId);
    tiling.set_eachCoreEndBlockId(endBlockId);

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

    uint32_t *seqOffset = tiling.get_seqOffset();
    uint32_t *startBlockId = tiling.get_eachCoreStartBlockId();
    uint32_t *endBlockId = tiling.get_eachCoreEndBlockId();

    OPS_LOG_D("seq offset:");
    for (auto i = 0; i < (tiling.get_batchSize() + 1); i++) {
        OPS_LOG_D("%d ", seqOffset[i]);
    }
    OPS_LOG_D("\n");

    OPS_LOG_D("core block range:\n");
    for (auto i = 0; i < MAX_AIV_NUM; i++) {
        OPS_LOG_E("", "core_id:%d startBlockId:%d endBlockId:%d\n", i, startBlockId[i], endBlockId[i]);
    }
}

}