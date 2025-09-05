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

#ifndef HSTU_SPLIT_CORE_POLICY_H
#define HSTU_SPLIT_CORE_POLICY_H

#include <unistd.h>

#include <cstdint>
#include <type_traits>
#include <numeric>
#include <memory>

#include "kernel_log.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;


namespace HstuDenseForward{
    struct BlockTaskInfo {
        uint32_t startBlockId = 0;
        uint32_t endBlockId = 0;
    };

    class BlockTaskAssign {
    public:
        BlockTaskAssign(uint32_t *seqOffsets,
                        uint32_t coreNum, uint32_t blockLen, uint32_t batchSize, uint32_t headNum)
        {
            this->seqOffsets = seqOffsets;
            this->coreNum = coreNum;
            this->blockLen = blockLen;
            this->batchSize = batchSize;
            this->headNum = headNum;
            int64_t blockNumberSize = headNum * batchSize * sizeof(int64_t);
            this->blockNumberGt.setGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(workspace) + GetBlockIdx() * blockNumberSize, blockNumberSize);
        }

        void PreInit(BlockTaskInfo *workTasks, int *workLoads) // 2408 * 8
        {
            // initialize workLoads to 0 for each core
            for (uint32_t i = 0; i < this->coreNum; i++) {
                workLoads[i] = 0;
            }

            // 得到每个batch 和 head的block个数
            for (auto batchId = 0; batchId < batchSize; batchId++) {
                auto batchBlockSize = this->seqOffsets[batchId + 1] - this->seqOffsets[batchId];

                for (auto headId = 0; headId < headNum; headId++) {
                    blockNumberGt[batchId * headNum + headId] =
                        (batchBlockSize + blockLen - 1) / blockLen;
                }
            }
        }

        bool BatchSwitch(
            uint32_t &batchId,
            uint32_t totalBatchSize,
            uint32_t &batchTaskNum)
        {
            if (blockNumberGt[batchId] == 0) {
                batchId++;
                if (batchId >= totalBatchSize) {
                    return false;
                }
                batchTaskNum = blockNumber[batchId];
            }
            return true;
        }

        void Compute(BlockTaskInfo *workTasks, int *workLoads, LocalTensor<int>&blockNumber, LocalTensor<int> &totalBlock)
        {
            // 得到每个batch 和 head的block个数
            uint32_t totalBatchSize = batchSize * headNum;
            PreInit(workTasks, workLoads);

            // 计算所有的task_num得到每个core 计算的task均值
            int64_t totalTaskNumber = 0;
            // 循环计算blockNumber
            uint32_t maxLen = 32 * 1024 / sizeof(int64_t);
            uint32_t loopLen = (totalBatchSize + maxLen - 1) / maxLen;
            int64_t totalBlockNumber = 0;

            for (uint32_t i = 0; i < loopLen; i++) {
                uint32_t copyLen = std::min(maxLen, totalBatchSize - i * maxLen);
                DataCopy(blockNumber, blockNumberGt, copyLen);
                Mul(totalBlock, blockNumber, blockNumber, copyLen);
                for (uint32_t j = 0; j < copyLen; j++) {
                    totalBlockNumber += totalBlock.GetValue(j);
                }
            }
            int64_t eachCoreTaskNumLimit = (totalTaskNumber + this->coreNum - 1) / this->coreNum;

            // 遍历workers 计算得到每一个works的任务量
            uint32_t batchId = 0;
            uint32_t batchTaskNum = blockNumberGt.GetValue(batchId);
            uint32_t processBlockNum = 0;
            uint32_t processTaskNum = 0;
            for (int i = 0; i < this->coreNum && batchId < totalBatchSize; i++) {
                BlockTaskInfo blockTask;
                blockTask.startBlockId = processBlockNum;

                while (workLoads[i] < eachCoreTaskNumLimit) {
                    workLoads[i] += batchTaskNum;
                    processTaskNum += batchTaskNum;
                    processBlockNum++;
                    blockNumberGt[batchId]--;
                    if (!BatchSwitch(batchId, totalBatchSize, batchTaskNum)) {
                        break;
                    }
                }

                blockTask.endBlockId = processBlockNum;
                workTasks[i] = blockTask;
            }
        }

        bool BatchSwitchCausal(
            uint32_t &batchId,
            uint32_t &taskNum,
            uint32_t totalBatchSize
        )
        {
            if (blockNumberGt[batchId] == 0) {
                batchId++;
                taskNum = 1;
                if (batchId >= totalBatchSize) {
                    return false;
                }
            }
            return true;
        }

        void ComputeCausal(BlockTaskInfo *workTasks, int *workLoads, LocalTensor<int64_t>&blockNumber, LocalTensor<int64_t> &totalBlock)
        {
            // 得到每个batch 和 head的block个数
            uint32_t totalBatchSize = batchSize * headNum;
            auto blockNumber = tmpBuff.AllocTensor<int>();
            PreInit(workTasks, workLoads, blockNumberGt);

            // 计算所有的task_num得到每个core 计算的task均值
            int64_t totalTaskNumber = 0;
            // 循环计算blockNumber maxLen 32K
            
            Adds(tmp, blockNumber, 1, totalBatchSize);
            Mul(totalBlock, tmp, blockNumber, totalBatchSize);
            ReduceSum(tmp, totalBlock, tmp, totalBatchSize); ///数据类型不支持
            uint32_t maxLen = 32 * 1024 / sizeof(int64_t);
            uint32_t loopLen = (totalBatchSize + maxLen - 1) / maxLen;

            for (uint32_t i = 0; i < loopLen; i++) {
                uint32_t copyLen = std::min(maxLen, totalBatchSize - i * maxLen);
                DataCopy(blockNumber, blockNumberGt, copyLen);
                Adds(totalBlock, blockNumber, 1, copyLen);
                Mul(totalBlock, totalBlock, blockNumber, copyLen);
                for (uint32_t j = 0; j < copyLen; j++) {
                    totalTaskNumber += totalBlock.GetValue(j);
                }
            }
            int64_t eachCoreTaskNumLimit = (totalTaskNumber + this->coreNum - 1) / this->coreNum;

            // 遍历workers 计算得到每一个works的任务量
            uint32_t batchId = 0;
            uint32_t taskNum = 1;
            uint32_t processBlockNum = 0;
            uint32_t processTaskNum = 0;
            for (int i = 0; i < this->coreNum && batchId < totalBatchSize; i++) {
                BlockTaskInfo blockTask;
                blockTask.startBlockId = processBlockNum;

                while (workLoads[i] < eachCoreTaskNumLimit) {
                    workLoads[i] += taskNum;
                    processTaskNum += taskNum;

                    taskNum++;
                    processBlockNum++;
                    blockNumberGt[batchId]--;
                    if (!BatchSwitchCausal(batchId, taskNum, totalBatchSize)) {
                        break;
                    }
                }

                blockTask.endBlockId = processBlockNum;
                workTasks[i] = blockTask;
            }
        }

    private:
        uint32_t *seqOffsets = nullptr;
        uint32_t coreNum = 0;
        uint32_t blockLen = 0;
        uint32_t batchSize = 0;
        uint32_t headNum = 0;
        GlobalTensor<int64_t>blockNumberGt;
    };
}
#endif