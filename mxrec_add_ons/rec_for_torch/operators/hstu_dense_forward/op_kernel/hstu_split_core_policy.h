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
        }

        void PreInit(
            std::vector<BlockTaskInfo> &workTasks, std::vector<int> &workLoads, std::vector<int64_t> &blockNumber)
        {
            workTasks.resize(this->coreNum);
            workLoads.resize(this->coreNum, 0);

            // 得到每个batch 和 head的block个数
            for (auto batchId = 0; batchId < batchSize; batchId++) {
                auto batchBlockSize = this->seqOffsets[batchId + 1] - this->seqOffsets[batchId];

                for (auto headId = 0; headId < headNum; headId++) {
                    blockNumber[batchId * headNum + headId] =
                        (batchBlockSize + blockLen - 1) / blockLen;
                }
            }
        }

        bool BatchSwitch(
            std::vector<int64_t> &blockNumber,
            uint32_t &batchId,
            uint32_t totalBatchSize,
            uint32_t &batchTaskNum)
        {
            if (blockNumber[batchId] == 0) {
                batchId++;
                if (batchId >= totalBatchSize) {
                    return false;
                }
                batchTaskNum = blockNumber[batchId];
            }
            return true;
        }

        void Compute(std::vector<BlockTaskInfo> &workTasks, std::vector<int> &workLoads)
        {
            // 得到每个batch 和 head的block个数
            uint32_t totalBatchSize = batchSize * headNum;
            std::vector<int64_t> blockNumber(totalBatchSize, 0);
            PreInit(workTasks, workLoads, blockNumber);

            // 计算所有的task_num得到每个core 计算的task均值
            int64_t totalTaskNumber = 0;
            totalTaskNumber = std::accumulate(blockNumber.begin(),
                                              blockNumber.end(),
                                              totalTaskNumber,
                                              [](int64_t val, int64_t x) {
                                                  return val + x * x;
                                              });

            int64_t eachCoreTaskNumLimit = (totalTaskNumber + this->coreNum - 1) / this->coreNum;

            // 遍历workers 计算得到每一个works的任务量
            uint32_t batchId = 0;
            uint32_t batchTaskNum = blockNumber[batchId];
            uint32_t processBlockNum = 0;
            uint32_t processTaskNum = 0;
            for (int i = 0; i < this->coreNum && batchId < totalBatchSize; i++) {
                BlockTaskInfo blockTask;
                blockTask.startBlockId = processBlockNum;

                while (workLoads[i] < eachCoreTaskNumLimit) {
                    workLoads[i] += batchTaskNum;
                    processTaskNum += batchTaskNum;
                    processBlockNum++;
                    blockNumber[batchId]--;
                    if (!BatchSwitch(blockNumber, batchId, totalBatchSize, batchTaskNum)) {
                        break;
                    }
                }

                blockTask.endBlockId = processBlockNum;
                workTasks[i] = blockTask;
            }
        }

        bool BatchSwitchCausal(
            std::vector<int64_t> &blockNumber,
            uint32_t &batchId,
            uint32_t &taskNum,
            uint32_t totalBatchSize
        )
        {
            if (blockNumber[batchId] == 0) {
                batchId++;
                taskNum = 1;
                if (batchId >= totalBatchSize) {
                    return false;
                }
            }
            return true;
        }

        void ComputeCausal(std::vector<BlockTaskInfo> &workTasks, std::vector<int> &workLoads)
        {
            // 得到每个batch 和 head的block个数
            uint32_t totalBatchSize = batchSize * headNum;
            std::vector<int64_t> blockNumber(totalBatchSize, 0);
            PreInit(workTasks, workLoads, blockNumber);

            // 计算所有的task_num得到每个core 计算的task均值
            int64_t totalTaskNumber = 0;
            totalTaskNumber = std::accumulate(blockNumber.begin(),
                                              blockNumber.end(),
                                              totalTaskNumber,
                                              [](int64_t val, int64_t x) {
                                                  return val + x * (x + 1) / CONST_2;
                                              });

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
                    blockNumber[batchId]--;
                    if (!BatchSwitchCausal(blockNumber, batchId, taskNum, totalBatchSize)) {
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
    };
}
#endif