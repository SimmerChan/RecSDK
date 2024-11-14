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

#ifndef LCCL_RMA_SWAP_H
#define LCCL_RMA_SWAP_H

#include "collectives.h"
using namespace AscendC;

constexpr uint64_t MAX_TABLE_NUM = 8;

/**
 * @brief updateTable是emb表的list，存的是各个表的地址
 */
#define RMA_SWAP_ARGS_FUN() \
GM_ADDR updateTable, GM_ADDR updateIndex, uint64_t updateLen, \
GM_ADDR svmBuffSwapIn, GM_ADDR svmBuffSwapOut, GM_ADDR usrWorkspace, int32_t dimNum, uint64_t *dimValue, GM_ADDR output

#define RMA_SWAP_ARGS_CALL() \
updateTable, updateIndex, updateLen, svmBuffSwapIn, svmBuffSwapOut, usrWorkspace, dimNum, dimValue, output


constexpr int32_t MAX_BLOCK_NUM = 8;

class RmaSwap : Collectives {
public:
    __aicore__ inline RmaSwap() : Collectives() {};

    __aicore__ inline void Init(RMA_SWAP_ARGS_FUN())
    {
        __ubuf__ uint64_t *ub_buff = (__ubuf__ uint64_t *)get_imm(0);
        Collectives::Init();
        dataHead.dataType = 0;
        dataHead.dimNum = dimNum;
        for (int i = 0; i < dimNum; ++i) {
            dataHead.dims[i] = dimValue[i]; // 当前只支持(length, emb_dim) float类型
        }
        embDim = dataHead.dims[1] * sizeof(float);
        uint64_t totalLength = updateLen * embDim;
        dataHead.totalLen = sizeof(RmaShmDataHead) + totalLength;
        dataHead.dataLen = totalLength;
        dataHead.readyLen = 0;
        this->updateTable = updateTable;
        this->updateIndex = updateIndex;
        this->updateLen = updateLen;
        this->svmBuffSwapIn = svmBuffSwapIn;
        this->svmBuffSwapOut = svmBuffSwapOut;
        this->usrWorkspace = usrWorkspace;
        this->output = output;

        swapFlagSwapIn = usrWorkspace + SWAP_IN_FLAG_OFFSET;
        swapFlagSwapOut = usrWorkspace + SWAP_OUT_FLAG_OFFSET;
        ClearFlag();

        cacheCapacity = SWAP_CACHE_SIZE / embDim;
        cacheFront = 0;
        cacheRear = 0;
    }
    /**
     * @brief 换入换出融合算子，至少需要4个core，换入换出分别2个，其中用1个core做SVM访问，剩下的core更新或查询HBM表
     * @tparam updateTable是要再HBM上置换的embedding表
     * @tparam updateIndex是置换embedding表项的索引
     * @tparam updateLen是置换表项的长度
     * @tparam svmBuffSwapIn换入SVM队列
     * @tparam svmBuffSwapOut换出SVM队列
     * @tparam usrWorkspace换入换出数据缓存，总22MB，1MB标志位+10MB数据缓存用于换入，1MB标志位+10MB数据缓存用于换出
     */
    __aicore__ inline void Process()
    {
        __ubuf__ uint64_t *ub_buff = (__ubuf__ uint64_t *)get_imm(0);
        const uint32_t processBlockNum = blockNum / 2;
        if (blockIdx < processBlockNum) {
            int32_t times = 0;
            do {
                ReadHeader(svmBuffSwapOut);
                if (queueHeader.seqIn - queueHeader.seqOut < queueHeader.queueCapacity) {
                    break;
                }
                if (times++ > RMA_BLOCKING_TIMES) {
//                    SetFlag(ub_buff, (__gm__ uint64_t *)output, 1);
                    return;
                }
            } while(true);
            dataHead.sequence = queueHeader.seqIn + 1;

            const uint32_t processBlockIdx = blockIdx % processBlockNum;
            emb_cache_swap = usrWorkspace + SWAP_OUT_CACHE_OFFSET;  // 换出缓存10MB
            // 换出
            if (processBlockIdx == 0) {
                OutfeedEnqueue(processBlockNum);
            } else {
                // 从updateTable表中
                LookUpTable(processBlockIdx, processBlockNum);
            }
        } else {
            // 换入
            int32_t times = 0;
            do {
                ReadHeader(svmBuffSwapIn);
                if ((queueHeader.seqIn - queueHeader.seqOut) > 0) {
                    // 队列非空
                    break;
                }
                // 队列为空时进行阻塞，解决host侧和device侧的读写时序问题
                if (times++ > RMA_BLOCKING_TIMES) {
//                    SetFlag(ub_buff, (__gm__ uint64_t *)output, 1);
                    return;
                }
            } while (true);
            dataHead.sequence = queueHeader.seqOut + 1;

            const uint32_t processBlockIdx = blockIdx % processBlockNum;
            emb_cache_swap = usrWorkspace + SWAP_IN_CACHE_OFFSET;    // 换入缓存10MB
            // 换出
            if (processBlockIdx == 0) {
                GetNext(processBlockNum);
            } else {
                UpdateTable(processBlockIdx, processBlockNum);
            }
        }
    }

    __aicore__ inline void LookUpTable(const uint32_t processBlockIdx, const uint32_t processBlockNum)
    {
        __ubuf__ uint64_t *ub_buff = (__ubuf__ uint64_t *)get_imm(0);
        __ubuf__ uint8_t *ub_data_buff = (__ubuf__ uint8_t *)get_imm(RMA_UB_DATA_BUFF_OFFSET);
        __gm__ uint64_t *outfeed_count = (__gm__ uint64_t *)(usrWorkspace + SWAP_OUT_FLAG_OFFSET);
        __gm__ uint64_t *lookup_flag = (__gm__ uint64_t *)(usrWorkspace + SWAP_OUT_FLAG_OFFSET + processBlockIdx * FLAG_UNIT_INT_NUM * sizeof(uint64_t));

        uint64_t outfeedCount = 0;
        uint64_t visitedIdx = processBlockIdx - 1;
        const uint64_t stride = processBlockNum - 1;
        while (visitedIdx < updateLen) {
            if (visitedIdx + stride - outfeedCount >= cacheCapacity - 1) {    // 队列满
                outfeedCount = GetFlag(ub_buff, outfeed_count);
//                outfeedCount = *outfeed_count;
                cacheFront = outfeedCount % cacheCapacity;
                continue;
            }
            uint64_t embIdx = *((__gm__ uint64_t *)updateIndex + visitedIdx);
            gm2ub(ub_data_buff, updateTable + embIdx * embDim, embDim);
            ub2gm(emb_cache_swap + cacheRear * embDim, ub_data_buff, embDim);
            cacheRear = (cacheRear + stride) % cacheCapacity;
            visitedIdx += stride;
            SetFlag(ub_buff, lookup_flag, visitedIdx);
        }
    }
    /**
     * @brief 用一个core做D2H拷贝，从emb_cache_swap拷贝到队列
     */
    __aicore__ inline void OutfeedEnqueue(const uint32_t processBlockNum)
    {
        __ubuf__ uint64_t *ub_buff = (__ubuf__ uint64_t *)get_imm(RMA_UB_B8_BUFF_OFFSET);
        __ubuf__ uint8_t *ub_data_buff = (__ubuf__ uint8_t *)get_imm(RMA_UB_DATA_BUFF_OFFSET);
        __gm__ uint64_t *outfeed_count = (__gm__ uint64_t *)(usrWorkspace + SWAP_OUT_FLAG_OFFSET);
        __gm__ uint64_t *seqInSwapOut = (__gm__ uint64_t *)svmBuffSwapOut + RmaQueueOffset::RMA_SEQ_IN_OFFSET;
        __gm__ uint64_t *tailSwapOut = (__gm__ uint64_t *)svmBuffSwapOut + RmaQueueOffset::RMA_QUEUE_TAIL_OFFSET;
        __gm__ uint64_t *buffLimitSwapOut = (__gm__ uint64_t *)svmBuffSwapOut + RmaQueueOffset::RMA_BUFF_LIMIT_OFFSET;
        GM_ADDR svmDataBuff;
        __gm__ uint64_t *svmReadyCount;

        // 生成入队数据的数据头信息
        __ubuf__ RmaShmDataHead *ub_datahead_buff = (__ubuf__ RmaShmDataHead *)get_imm(0);  // 数据头
        ub_datahead_buff->totalLen = dataHead.totalLen;
        ub_datahead_buff->sequence = dataHead.sequence;
        ub_datahead_buff->dataType = dataHead.dataType;
        ub_datahead_buff->dimNum = dataHead.dimNum;
        for (int i = 0; i < dataHead.dimNum; ++i) {
            ub_datahead_buff->dims[i] = dataHead.dims[i] ;
        }
        ub_datahead_buff->dataLen = dataHead.dataLen;
        ub_datahead_buff->readyLen = dataHead.readyLen;
        pipe_barrier(PIPE_ALL);

        // 如果队列尾部的空余放不下新插入的数据，则从队列头部插入，当前约束队列不够大，不会被写满
        if (queueHeader.tailOffset + ub_datahead_buff->totalLen > queueHeader.totalMemSize) {
            *ub_buff = queueHeader.tailOffset;
            CpUB2GM<uint64_t>(buffLimitSwapOut, ub_buff, sizeof(uint64_t));

            // 从队列头的位置，把数据头信息写入到shm buffer中
            ub2gm(svmBuffSwapOut + RMA_SHM_HEAD_LEN, (__ubuf__ uint8_t *)ub_datahead_buff, RMA_SHM_DATA_HEAD);
            svmDataBuff = svmBuffSwapOut + RMA_SHM_HEAD_LEN + RMA_SHM_DATA_HEAD;
            svmReadyCount = (__gm__ uint64_t *)(svmBuffSwapOut + RMA_SHM_HEAD_LEN) + RMA_READY_LEN_OFFSET;
            *ub_buff = RMA_SHM_HEAD_LEN + ub_datahead_buff->totalLen;
            CpUB2GM<uint64_t>(tailSwapOut, ub_buff, sizeof(uint64_t));
        } else {
            // 正常逻辑，从队列写入，把数据头信息写入到shm buffer中
            ub2gm(svmBuffSwapOut + queueHeader.tailOffset, (__ubuf__ uint8_t *)ub_datahead_buff, RMA_SHM_DATA_HEAD);
            svmDataBuff = svmBuffSwapOut + queueHeader.tailOffset + RMA_SHM_DATA_HEAD;
            svmReadyCount = (__gm__ uint64_t *)(svmBuffSwapOut + queueHeader.tailOffset) + RMA_READY_LEN_OFFSET;
            *ub_buff = queueHeader.tailOffset + ub_datahead_buff->totalLen;
            CpUB2GM<uint64_t>(tailSwapOut, ub_buff, sizeof(uint64_t));
        }

        __gm__ uint64_t *lookUpFlags[MAX_BLOCK_NUM];
        for (int i = 1; i < processBlockNum; ++i) {
            lookUpFlags[i - 1] = (__gm__ uint64_t *)swapFlagSwapOut + i * FLAG_UNIT_INT_NUM;
        }
        uint64_t lookUpCount = 0;
        uint64_t swapOutCount = 0;
        while (swapOutCount < updateLen) {
            if (lookUpCount <= swapOutCount) {  // 队列空
                lookUpCount = GetMinFlag(ub_buff, lookUpFlags, processBlockNum - 1);
                cacheRear = lookUpCount % cacheCapacity;
                continue;
            }
            uint64_t copyCount = (cacheCapacity + cacheRear - cacheFront) % cacheCapacity;
            if (cacheFront > cacheRear) {  // 跨末尾，地址不连续
                copyCount = cacheCapacity - cacheFront;
            }
            gm2gm(copyCount * dataHead.dims[1] * sizeof(float), ub_data_buff,
                  svmDataBuff + swapOutCount * embDim, emb_cache_swap + cacheFront * embDim);
            cacheFront = (cacheFront + copyCount) % cacheCapacity;
            swapOutCount += copyCount;
            SetFlag(ub_buff, outfeed_count, swapOutCount);
            SetFlag(ub_buff, svmReadyCount, swapOutCount);
        }

        // 更新seqIn
        *ub_buff = dataHead.sequence;
        CpUB2GM<uint64_t>(seqInSwapOut, ub_buff, sizeof(uint64_t));
    }

    __aicore__ inline void UpdateTable(const uint32_t processBlockIdx, const uint32_t processBlockNum)
    {
        __ubuf__ uint64_t *ub_buff = (__ubuf__ uint64_t *)get_imm(RMA_UB_B8_BUFF_OFFSET);
        __ubuf__ uint8_t *ub_data_buff = (__ubuf__ uint8_t *)get_imm(RMA_UB_DATA_BUFF_OFFSET);
        __gm__ uint64_t *outfeed_count = (__gm__ uint64_t *)(usrWorkspace + SWAP_OUT_FLAG_OFFSET);
        __gm__ uint64_t *getnext_count = (__gm__ uint64_t *)(usrWorkspace + SWAP_IN_FLAG_OFFSET);
        __gm__ uint64_t *update_flag = (__gm__ uint64_t *)(usrWorkspace + SWAP_IN_FLAG_OFFSET) + processBlockIdx * FLAG_UNIT_INT_NUM;

        __gm__ uint64_t *lookUpFlags[MAX_BLOCK_NUM];  // 换出查表的标志位
        for (int i = 1; i < processBlockNum; ++i) {
            lookUpFlags[i - 1] = (__gm__ uint64_t *)swapFlagSwapOut + i * FLAG_UNIT_INT_NUM;
        }

        uint64_t getnextCount = 0;  // 已经从shm读进来的emb数
        uint64_t lookUpCount = 0;   // 已经换出的emb数（避免换入的新数据覆盖掉旧数据）
        uint64_t visitedIdx = processBlockIdx - 1;  // 当前要更新的emb索引
        const uint64_t stride = processBlockNum - 1;    // 多core时每个core的步长

        while (visitedIdx < updateLen) {
            if (getnextCount <= visitedIdx || visitedIdx >= lookUpCount) {    // 缓存队列空 或者 数据还没换出
                getnextCount = GetFlag(ub_buff, getnext_count);
//                getnextCount = *getnext_count;
                cacheRear = getnextCount % cacheCapacity;
                lookUpCount = GetMinFlag(ub_buff, lookUpFlags, processBlockNum - 1);
                continue;
            }
            uint64_t embIdx = *((__gm__ uint64_t *)updateIndex + visitedIdx);
            gm2ub(ub_data_buff, emb_cache_swap + cacheFront * embDim, embDim);
            ub2gm(updateTable + embIdx * embDim, ub_data_buff, embDim);
            cacheFront = (visitedIdx + 1) % cacheCapacity;

            SetFlag(ub_buff, update_flag, visitedIdx + 1);
            visitedIdx += stride;

//            SetFlag(ub_buff, (__gm__ uint64_t *)output + blockIdx, getnextCount);
        }
    }

    __aicore__ inline void GetNext(const uint32_t processBlockNum)
    {
        __ubuf__ RmaShmDataHead *ub_datahead_buff = (__ubuf__ RmaShmDataHead *)get_imm(0);  // 数据头
        __ubuf__ uint64_t *ub_buff = (__ubuf__ uint64_t *)get_imm(RMA_UB_B8_BUFF_OFFSET);
        __ubuf__ uint8_t *ub_data_buff = (__ubuf__ uint8_t *)get_imm(RMA_UB_DATA_BUFF_OFFSET);
        __gm__ uint64_t *getnext_count = (__gm__ uint64_t *)(usrWorkspace + SWAP_IN_FLAG_OFFSET);

        __gm__ uint64_t *seqOutSwapIn = (__gm__ uint64_t *)svmBuffSwapIn + RmaQueueOffset::RMA_SEQ_OUT_OFFSET;
        __gm__ uint64_t *frontSwapIn = (__gm__ uint64_t *)svmBuffSwapIn + RmaQueueOffset::RMA_QUEUE_FRONT_OFFSET;
        __gm__ uint64_t *buffLimitSwapIn = (__gm__ uint64_t *)svmBuffSwapIn + RmaQueueOffset::RMA_BUFF_LIMIT_OFFSET;

        uint64_t frontOffset = queueHeader.frontOffset;
        if (queueHeader.buffLimit == frontOffset) {
            // 队列尾部的数据都已读取完，需要返回到首部
            frontOffset = RMA_SHM_HEAD_LEN;
            *ub_buff = frontOffset;
            CpUB2GM<uint64_t>(frontSwapIn, ub_buff, sizeof(uint64_t));

            // 初始化队列的buffer limit
            *ub_buff = 0;
            CpUB2GM<uint64_t>(buffLimitSwapIn, ub_buff, sizeof(uint64_t));
        }
        CpGM2UB<uint8_t>((__ubuf__ uint8_t *)ub_datahead_buff, svmBuffSwapIn + frontOffset, RMA_SHM_DATA_HEAD);
        const uint64_t sizeOfTotalData = ub_datahead_buff->totalLen;
        const uint64_t sizeOfData = sizeOfTotalData - RMA_SHM_DATA_HEAD; // 减去数据头RMA_SHM_DATA_HEAD
        uint64_t sequence = ub_datahead_buff->sequence;
        GM_ADDR svmDataBuff = svmBuffSwapIn + frontOffset + RMA_SHM_DATA_HEAD;
        __gm__ uint64_t *ready_len = (__gm__ uint64_t *)(svmBuffSwapIn + frontOffset) + RMA_READY_LEN_OFFSET;
        pipe_barrier(PIPE_ALL);

        __gm__ uint64_t *updateFlags[MAX_BLOCK_NUM];
        for (int i = 1; i < processBlockNum; ++i) {
            updateFlags[i - 1] = (__gm__ uint64_t *)swapFlagSwapIn + i * FLAG_UNIT_INT_NUM;
        }

        if (sizeOfData > 0) {
            uint64_t updateCount = 0;   // 单位embDim
            uint64_t readyLen = 0;      // 单位字节
            uint64_t copyOffset = 0;    // 单位字节
            while (copyOffset < sizeOfData) {
                if (readyLen <= copyOffset || (cacheRear + 1) % cacheCapacity == cacheFront) {
                    readyLen = GetFlag(ub_buff, ready_len);
                    updateCount = GetMinFlag(ub_buff, updateFlags, processBlockNum - 1);
                    cacheFront = updateCount % cacheCapacity;
                    continue;
                }
                uint64_t copySize = readyLen - copyOffset;
                uint64_t cacheSize = 0;

                if (cacheRear >= cacheFront) {
                    if (cacheFront == 0) {
                        cacheSize = (cacheCapacity - cacheRear - 1) * embDim;
                    } else {
                        cacheSize = (cacheCapacity - cacheRear) * embDim;
                    }
                } else {
                    cacheSize = (cacheFront - cacheRear - 1) * embDim;
                }
                copySize = (copySize > cacheSize) ? cacheSize : copySize;
                gm2gm(copySize, ub_data_buff, emb_cache_swap + cacheRear * embDim, svmDataBuff + copyOffset);
                copyOffset += copySize;
                cacheRear = (copyOffset / embDim) % cacheCapacity;
                SetFlag(ub_buff, getnext_count, copyOffset / embDim);
//                SetFlag(ub_buff, (__gm__ uint64_t *)output + blockIdx, copyOffset / embDim);
            }
        }
        // 更新队头offset
        *ub_buff = frontOffset + sizeOfTotalData;
        CpUB2GM<uint64_t>(frontSwapIn, ub_buff, sizeof(uint64_t));
        // 更新seqOut
        *ub_buff = sequence;
        CpUB2GM<uint64_t>(seqOutSwapIn, ub_buff, sizeof(uint64_t));
    }
private:
    __aicore__ inline void ReadHeader(GM_ADDR svm_buff)
    {
        __ubuf__ RmaShmHeader *ub_buff = (__ubuf__ RmaShmHeader *)get_imm(0);

        // 读取SwapOut队列的队列头
        CpGM2UB<RmaShmHeader>(ub_buff, (__gm__ RmaShmHeader *)svm_buff, sizeof(RmaShmHeader));
        // 读取最新写入的seq
        queueHeader.queueCapacity = ub_buff->queueCapacity;
        queueHeader.totalMemSize = ub_buff->totalMemSize;
        queueHeader.seqIn = ub_buff->seqIn;
        queueHeader.seqOut = ub_buff->seqOut;
        queueHeader.frontOffset = ub_buff->frontOffset;
        queueHeader.tailOffset = ub_buff->tailOffset;
        queueHeader.buffLimit = ub_buff->buffLimit;

        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void ClearFlag()
    {
        if (blockIdx == 0) {
            __ubuf__ uint64_t *ub_buff = (__ubuf__ uint64_t *)get_imm(0);
            for (int i = 0; i < MAX_BLOCK_NUM * FLAG_UNIT_INT_NUM; ++i) {
                *(ub_buff + i) = 0;
            }
            CpUB2GM<uint8_t>(swapFlagSwapIn, (__ubuf__ uint8_t *)ub_buff, MAX_BLOCK_NUM * FLAG_UNIT_INT_NUM * sizeof(uint64_t));
            CpUB2GM<uint8_t>(swapFlagSwapOut, (__ubuf__ uint8_t *)ub_buff, MAX_BLOCK_NUM * FLAG_UNIT_INT_NUM * sizeof(uint64_t));
        }
    }

private:
    RmaShmHeader queueHeader;
    RmaShmDataHead dataHead;
    GM_ADDR updateTable;
    GM_ADDR updateIndex;
    uint64_t updateLen;
    GM_ADDR svmBuffSwapIn;
    GM_ADDR svmBuffSwapOut;
    GM_ADDR usrWorkspace;
    GM_ADDR swapFlagSwapIn;
    GM_ADDR swapFlagSwapOut;
    uint64_t embDim;        // 每条emb的维度，单位字节
    GM_ADDR emb_cache_swap;
    uint64_t cacheCapacity; // 数据缓存能容纳的emb条数
    uint64_t cacheFront;    // 数据缓存队列头索引
    uint64_t cacheRear;     // 数据缓存队列尾索引
    GM_ADDR output;         // 算子返回值
};

#endif //LCCL_RMA_SWAP_H
