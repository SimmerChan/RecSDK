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

#ifndef LCCL_RMA_SWAP_MULTI_TABLES_H
#define LCCL_RMA_SWAP_MULTI_TABLES_H

#include "collectives.h"
using namespace AscendC;

constexpr uint64_t MAX_TABLE_NUM = 6;
constexpr uint64_t GET_NEXT_THREAD_NUM = 4;

/**
 * @brief updateTables是emb表的list，存的是各个表的地址
 */
#define RMA_SWAP_MULTI_TABLE_ARGS_FUN() \
GM_ADDR table_a, GM_ADDR table_b, GM_ADDR table_c, GM_ADDR table_d, GM_ADDR table_e, GM_ADDR table_f, \
int tableNum, int tableLength, GM_ADDR swapInIndex, GM_ADDR swapOutIndex, uint64_t swapInLen, \
GM_ADDR svmBuffSwapIn, GM_ADDR svmBuffSwapOut, GM_ADDR usrWorkspace, int32_t dimNum, uint64_t *dimValue, GM_ADDR output

#define RMA_SWAP_MULTI_TABLE_ARGS_CALL() \
table_a, table_b, table_c, table_d, table_e, table_f, \
tableNum, tableLength, swapInIndex, swapOutIndex, swapInLen, \
svmBuffSwapIn, svmBuffSwapOut, usrWorkspace, dimNum, dimValue, output


class RmaSwapMultiTables : Collectives {
public:
    __aicore__ inline RmaSwapMultiTables() : Collectives() {};

    __aicore__ inline void Init(RMA_SWAP_MULTI_TABLE_ARGS_FUN())
    {
        __ubuf__ uint64_t *ub_buff = (__ubuf__ uint64_t *)get_imm(0);
        Collectives::Init();

        this->tableNum = tableNum;
        this->updateTables[0] = table_a;
        this->updateTables[1] = table_b;
        this->updateTables[2] = table_c;
        this->updateTables[3] = table_d;
        this->updateTables[4] = table_e;
        this->updateTables[5] = table_f;
        this->swapInIndex = swapInIndex;
        this->swapOutIndex = swapOutIndex;
        this->swapInLen = swapInLen;
        this->swapOutLen = dimValue[0];
        this->svmBuffSwapIn = svmBuffSwapIn;
        this->svmBuffSwapOut = svmBuffSwapOut;
        this->usrWorkspace = usrWorkspace;
        this->output = output;

        dataHeadSwapOut.dataType = 0;
        dataHeadSwapOut.dimNum = dimNum;
        for (int i = 0; i < dimNum - 1; ++i) {
            dataHeadSwapOut.dims[i] = dimValue[i]; // 当前只支持(length, emb_dim) float类型
        }
        dataHeadSwapOut.dims[dimNum - 1] = dimValue[dimNum - 1] * tableNum;
        embDim = dataHeadSwapOut.dims[1] * sizeof(float);  // rma上每条emb的维度
        embDimSplit = dimValue[dimNum - 1] * sizeof(float);       // 每个emb的维度，暂时认为都是相同的
        uint64_t totalLength = swapOutLen * embDim;
        dataHeadSwapOut.totalLen = totalLength + RMA_SHM_DATA_HEAD;
        dataHeadSwapOut.dataLen = totalLength;
        dataHeadSwapOut.readyLen = 0;

        swapFlagSwapIn = usrWorkspace + SWAP_IN_FLAG_OFFSET;
        swapFlagSwapOut = usrWorkspace + SWAP_OUT_FLAG_OFFSET;

        processBlockNum = blockNum / 2;
        processBlockIdx = blockIdx % processBlockNum;

        GetQueHead();

        cacheCapacity = SWAP_CACHE_SIZE / embDim;
        cacheFront = 0;
        cacheRear = 0;

        SyncPreprocess();
    }
    /**
     * @brief 换入换出融合算子，至少需要4个core，换入换出分别2个，其中用1个core做SVM访问，剩下的core更新或查询表
     * @tparam updateTables是要再上置换的embedding表
     * @tparam swapInIndex是置换embedding表项的索引
     * @tparam swapInLen是置换表项的长度
     * @tparam svmBuffSwapIn换入SVM队列
     * @tparam svmBuffSwapOut换出SVM队列
     * @tparam usrWorkspace换入换出数据缓存，总22MB，1MB标志位+10MB数据缓存用于换入，1MB标志位+10MB数据缓存用于换出
     */
    __aicore__ inline void Process()
    {
        __ubuf__ uint64_t *ub_buff = (__ubuf__ uint64_t *)get_imm(0);
        if (GetFlag<uint64_t>(ub_buff, (__gm__ uint64_t *)output + blockIdx) != 0) {
            SyncPostprocess();
            return;
        }
        if (blockIdx < processBlockNum) {   // 换出
            if (processBlockIdx == 0) {
                OutfeedEnqueue();
            } else {
                // 从updateTables表中
                LookUpTable();
            }
        } else {    // 换入
            if (processBlockIdx < GET_NEXT_THREAD_NUM) {
                GetNextMultiThreads();
            } else {
                UpdateTable();
            }
        }
        SyncPostprocess();
    }
private:
    __aicore__ inline bool Full(uint64_t dataSize)
    {
        dataSize += RMA_SHM_DATA_HEAD;
        if (queueHeader.seqIn - queueHeader.seqOut >= queueHeader.queueCapacity) {
            return true;
        }
        if (queueHeader.tailOffset + dataSize > queueHeader.totalMemSize) {
            if (dataSize + RMA_SHM_HEAD_LEN > queueHeader.frontOffset) {
                return true;
            }
        } else {
            if (queueHeader.tailOffset < queueHeader.frontOffset &&
                        queueHeader.tailOffset + dataSize >= queueHeader.frontOffset) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline void GetQueHead()
    {
        __ubuf__ uint64_t *ub_buff = (__ubuf__ uint64_t *)get_imm(0);
        SetFlag(ub_buff, (__gm__ uint64_t *)output + blockIdx, 0);
        uint64_t times = 0;
        if (blockIdx < processBlockNum) {   // 换出
            do {
                ReadHeader(svmBuffSwapOut);
                if (!Full(dataHeadSwapOut.dataLen)) {
                    break;
                }
                if (++times > TIME_OUT) {
                    SetFlag(ub_buff, (__gm__ uint64_t *)output + blockIdx, 10000);
                    return;
                }
            } while(true);
            dataHeadSwapOut.sequence = queueHeader.seqIn + 1;
            embSwapCache = usrWorkspace + SWAP_OUT_CACHE_OFFSET;  // 换出缓存10MB
        } else {    // 换入
            do {
                ReadHeader(svmBuffSwapIn);
                if ((queueHeader.seqIn - queueHeader.seqOut) > 0) {
                    // 队列非空
                    break;
                }
                if (++times > TIME_OUT) {
                    SetFlag(ub_buff, (__gm__ uint64_t *)output + blockIdx, 10000);
                    return;
                }
                // 队列为空时进行阻塞，解决host侧和device侧的读写时序问题
            } while (true);
            embSwapCache = usrWorkspace + SWAP_IN_CACHE_OFFSET;    // 换入缓存10MB
        }
    }

    __aicore__ inline void LookUpTable()
    {
        __ubuf__ uint64_t *ub_buff = (__ubuf__ uint64_t *)get_imm(0);
        __ubuf__ uint8_t *ub_data_buff = (__ubuf__ uint8_t *)get_imm(RMA_UB_DATA_BUFF_OFFSET);
        __gm__ uint64_t *outfeed_count = (__gm__ uint64_t *)swapFlagSwapOut;
        __gm__ uint64_t *lookup_flag = (__gm__ uint64_t *)swapFlagSwapOut + processBlockIdx * FLAG_UNIT_INT_NUM;

        uint64_t outfeedCount = 0;
        uint64_t visitedIdx = processBlockIdx - 1;
        const uint64_t stride = processBlockNum - 1;
        cacheRear = visitedIdx % cacheCapacity;
        uint64_t loopCount = 0;
        while (visitedIdx < swapOutLen) {
            if (visitedIdx + 1 - outfeedCount >= cacheCapacity - 1) {    // 队列满
                outfeedCount = GetFlag2(ub_buff, outfeed_count);
                continue;
            }
            uint64_t embIdx = *((__gm__ uint64_t *)swapOutIndex + visitedIdx);
            for (int t = 0; t < tableNum; ++t) {
                gm2gm(embDimSplit, ub_data_buff, embSwapCache + cacheRear * embDim + t * embDimSplit,
                      updateTables[t] + embIdx * embDimSplit);
            }
            cacheRear = (cacheRear + stride) % cacheCapacity;
            visitedIdx += stride;
            if (loopCount % 8 == 0 || visitedIdx + 1 - outfeedCount >= cacheCapacity - 1) {
                SetFlag(ub_buff, lookup_flag, visitedIdx);
            }
            loopCount++;
        }
        SetFlag(ub_buff, lookup_flag, visitedIdx);
    }
    /**
     * @brief 用一个core做D2H拷贝，从embSwapCache拷贝到队列
     */
    __aicore__ inline void OutfeedEnqueue()
    {
        __ubuf__ uint64_t *ub_buff = (__ubuf__ uint64_t *)get_imm(RMA_UB_B8_BUFF_OFFSET);
        __ubuf__ uint8_t *ub_data_buff = (__ubuf__ uint8_t *)get_imm(RMA_UB_DATA_BUFF_OFFSET);
        __gm__ uint64_t *outfeed_count = (__gm__ uint64_t *)swapFlagSwapOut;
        __gm__ uint64_t *seqInSwapOut = (__gm__ uint64_t *)svmBuffSwapOut + RmaQueueOffset::RMA_SEQ_IN_OFFSET;
        __gm__ uint64_t *tailSwapOut = (__gm__ uint64_t *)svmBuffSwapOut + RmaQueueOffset::RMA_QUEUE_TAIL_OFFSET;
        __gm__ uint64_t *buffLimitSwapOut = (__gm__ uint64_t *)svmBuffSwapOut + RmaQueueOffset::RMA_BUFF_LIMIT_OFFSET;
        GM_ADDR svmDataBuff;
        __gm__ uint64_t *svmReadyCount;

        // 生成入队数据的数据头信息
        __ubuf__ RmaShmDataHead *ub_datahead_buff = (__ubuf__ RmaShmDataHead *)get_imm(0);  // 数据头
        ub_datahead_buff->totalLen = dataHeadSwapOut.totalLen;
        ub_datahead_buff->sequence = dataHeadSwapOut.sequence;
        ub_datahead_buff->dataType = dataHeadSwapOut.dataType;
        ub_datahead_buff->dimNum = dataHeadSwapOut.dimNum;
        for (int i = 0; i < dataHeadSwapOut.dimNum; ++i) {
            ub_datahead_buff->dims[i] = dataHeadSwapOut.dims[i] ;
        }
        ub_datahead_buff->dataLen = dataHeadSwapOut.dataLen;
        ub_datahead_buff->readyLen = dataHeadSwapOut.readyLen;
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
        pipe_barrier(PIPE_ALL);
        uint64_t lookUpCount = 0;
        uint64_t swapOutCount = 0;
        uint64_t loopCount = 0;
        while (swapOutCount < swapOutLen) {
            if (lookUpCount <= swapOutCount) {  // 队列空
                lookUpCount = GetMinFlag(ub_buff, lookUpFlags, processBlockNum - 1);
                cacheRear = lookUpCount % cacheCapacity;
                continue;
            }
            uint64_t copyCount = (cacheCapacity + cacheRear - cacheFront) % cacheCapacity;
            if (cacheFront > cacheRear) {  // 跨末尾，地址不连续
                copyCount = cacheCapacity - cacheFront;
            }
            gm2gm(copyCount * dataHeadSwapOut.dims[1] * sizeof(float), ub_data_buff,
                  svmDataBuff + swapOutCount * embDim, embSwapCache + cacheFront * embDim);
            cacheFront = (cacheFront + copyCount) % cacheCapacity;
            swapOutCount += copyCount;
            if (loopCount % 4 == 0 || lookUpCount <= swapOutCount) {
                SetFlag(ub_buff, outfeed_count, swapOutCount);
            }
            loopCount++;
        }
        SetFlag(ub_buff, outfeed_count, swapOutCount);

        // 更新seqIn
        *ub_buff = dataHeadSwapOut.sequence;
        CpUB2GM<uint64_t>(seqInSwapOut, ub_buff, sizeof(uint64_t));
    }

    __aicore__ inline void UpdateTable()
    {
        __ubuf__ uint64_t *ub_buff = (__ubuf__ uint64_t *)get_imm(RMA_UB_B8_BUFF_OFFSET);
        __ubuf__ uint8_t *ub_data_buff = (__ubuf__ uint8_t *)get_imm(RMA_UB_DATA_BUFF_OFFSET);
        __gm__ uint64_t *getnext_count = (__gm__ uint64_t *)swapFlagSwapIn;
        __gm__ uint64_t *update_flag = (__gm__ uint64_t *)swapFlagSwapIn + processBlockIdx * FLAG_UNIT_INT_NUM;

        __gm__ uint64_t *lookUpFlags[MAX_BLOCK_NUM];  // 换出查表的标志位
        for (int i = 1; i < processBlockNum; ++i) {
            lookUpFlags[i - 1] = (__gm__ uint64_t *)swapFlagSwapOut + i * FLAG_UNIT_INT_NUM;
        }

        __gm__ uint64_t *getnextFlags[MAX_BLOCK_NUM];  // 换出查表的标志位
        for (int i = 0; i < GET_NEXT_THREAD_NUM; ++i) {
            getnextFlags[i] = (__gm__ uint64_t *)swapFlagSwapIn + i * FLAG_UNIT_INT_NUM;
        }

        uint64_t getnextCount = 0;  // 已经从shm读进来的emb数
        const uint64_t freeCount = swapInLen - swapOutLen;  // emb表内空闲的emb个数
        uint64_t lookUpCount = 0;   // 已经换出的emb数（避免换入的新数据覆盖掉旧数据）
        uint64_t visitedIdx = processBlockIdx - GET_NEXT_THREAD_NUM;  // 当前要更新的emb索引
        const uint64_t stride = processBlockNum - GET_NEXT_THREAD_NUM;    // 多core时每个core的步长
        cacheFront = visitedIdx % cacheCapacity;
        uint64_t loopCount = 0;
        while (visitedIdx < swapInLen) {
            if (getnextCount <= visitedIdx || visitedIdx >= freeCount + lookUpCount) {    // 缓存队列空 或者 数据还没换出
                if (getnextCount < swapInLen) {
                    getnextCount = GetMinFlag(ub_buff, getnextFlags, GET_NEXT_THREAD_NUM);
                }
                if (lookUpCount < swapOutLen) {
                    lookUpCount = GetMinFlag(ub_buff, lookUpFlags, processBlockNum - 1);
                }
                continue;
            }
            uint64_t embIdx = *((__gm__ uint64_t *)swapInIndex + visitedIdx);
            for (int t = 0; t < tableNum; ++t) {
                gm2gm(embDimSplit, ub_data_buff, updateTables[t] + embIdx * embDimSplit, embSwapCache + cacheFront * embDim + t * embDimSplit);
            }

            visitedIdx += stride;
            cacheFront = visitedIdx % cacheCapacity;
            if (loopCount % 8 == 0 || getnextCount <= visitedIdx) {
                SetFlag(ub_buff, update_flag, visitedIdx);
            }
            loopCount++;
        }
        SetFlag(ub_buff, update_flag, visitedIdx);
    }

    __aicore__ inline void GetNextMultiThreads()
    {
        __ubuf__ RmaShmDataHead *ub_datahead_buff = (__ubuf__ RmaShmDataHead *)get_imm(0);  // 数据头
        __ubuf__ uint64_t *ub_buff = (__ubuf__ uint64_t *)get_imm(RMA_UB_B8_BUFF_OFFSET);
        __ubuf__ uint8_t *ub_data_buff = (__ubuf__ uint8_t *)get_imm(RMA_UB_DATA_BUFF_OFFSET);
        __gm__ uint64_t *getnext_count = (__gm__ uint64_t *)swapFlagSwapIn + processBlockIdx * FLAG_UNIT_INT_NUM;

        __gm__ uint64_t *seqOutSwapIn = (__gm__ uint64_t *)svmBuffSwapIn + RmaQueueOffset::RMA_SEQ_OUT_OFFSET;
        __gm__ uint64_t *frontSwapIn = (__gm__ uint64_t *)svmBuffSwapIn + RmaQueueOffset::RMA_QUEUE_FRONT_OFFSET;
        __gm__ uint64_t *buffLimitSwapIn = (__gm__ uint64_t *)svmBuffSwapIn + RmaQueueOffset::RMA_BUFF_LIMIT_OFFSET;

        bool updataBuffLimit = false;
        uint64_t frontOffset = queueHeader.frontOffset;
        if (queueHeader.buffLimit == frontOffset) {
            // 队列尾部的数据都已读取完，需要返回到首部
            frontOffset = RMA_SHM_HEAD_LEN;
            updataBuffLimit = true;
        }

        CpGM2UB<uint8_t>((__ubuf__ uint8_t *)ub_datahead_buff, svmBuffSwapIn + frontOffset, RMA_SHM_DATA_HEAD);
        const uint64_t sizeOfTotalData = ub_datahead_buff->totalLen;
        const uint64_t sizeOfData = ub_datahead_buff->dataLen;
        const uint64_t sequence = ub_datahead_buff->sequence;
        GM_ADDR svmDataBuff = svmBuffSwapIn + frontOffset + RMA_SHM_DATA_HEAD;
        __gm__ uint64_t *ready_len = (__gm__ uint64_t *)(svmBuffSwapIn + frontOffset) + RMA_READY_LEN_OFFSET;
        pipe_barrier(PIPE_ALL);

        __gm__ uint64_t *updateFlags[MAX_BLOCK_NUM];
        for (int i = GET_NEXT_THREAD_NUM; i < processBlockNum; ++i) {
            updateFlags[i - GET_NEXT_THREAD_NUM] = (__gm__ uint64_t *)swapFlagSwapIn + i * FLAG_UNIT_INT_NUM;
        }
        const uint64_t pipeBlockSize = ((2 * UNIT_COPY_SIZE) / embDim) * embDim;    // 777216
        const uint64_t stride = pipeBlockSize * GET_NEXT_THREAD_NUM;

        if (sizeOfData > 0) {
            uint64_t updateCount = 0;   // 单位embDim
            uint64_t getnextCount = 0;  // 单位embDim
            uint64_t readyLen = 0; // 单位字节
            uint64_t copyOffset = processBlockIdx * pipeBlockSize;    // 单位字节
            cacheRear = (copyOffset / embDim) % cacheCapacity;
            while (copyOffset < sizeOfData) {
                if ((readyLen < sizeOfData && readyLen < copyOffset + pipeBlockSize) || (getnextCount >= updateCount && getnextCount - updateCount > cacheCapacity)) {
                    if (readyLen < sizeOfData) {
                        readyLen = GetFlag2(ub_buff, ready_len);
                    }
                    updateCount = GetMinFlag(ub_buff, updateFlags, processBlockNum - GET_NEXT_THREAD_NUM);
                    cacheFront = updateCount % cacheCapacity;
                    continue;
                }
                uint64_t copySize = (copyOffset + pipeBlockSize <= sizeOfData) ? pipeBlockSize : (sizeOfData - copyOffset);
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
                gm2gm(copySize, ub_data_buff, embSwapCache + cacheRear * embDim, svmDataBuff + copyOffset);   // 影响性能

                if (copyOffset + stride >= sizeOfData) {
                    copyOffset = sizeOfData;
                } else {
                    copyOffset += stride;
                }
                getnextCount = copyOffset / embDim;
                cacheRear = getnextCount % cacheCapacity;
                SetFlag(ub_buff, getnext_count, getnextCount);
            }
            SetFlag(ub_buff, getnext_count, getnextCount);
        }
        if (processBlockIdx == 0) {
            __gm__ uint64_t *getnextFlags[MAX_BLOCK_NUM];
            for (int i = 1; i < GET_NEXT_THREAD_NUM; ++i) {
                getnextFlags[i - 1] = (__gm__ uint64_t *)swapFlagSwapIn + i * FLAG_UNIT_INT_NUM;
            }
            uint64_t minGetnext = 0;
            while (minGetnext < swapInLen) {
                minGetnext = GetMinFlag(ub_buff, getnextFlags, GET_NEXT_THREAD_NUM - 1);
            }

            if (updataBuffLimit) {
                // 队列尾部的数据都已读取完，需要返回到首部
                // 初始化队列的buffer limit
                *ub_buff = 0;
                CpUB2GM<uint64_t>(buffLimitSwapIn, ub_buff, sizeof(uint64_t));
            }

            // 更新队头offset
            *ub_buff = frontOffset + sizeOfTotalData;
            CpUB2GM<uint64_t>(frontSwapIn, ub_buff, sizeof(uint64_t));
            // 更新seqOut
            *ub_buff = sequence;
            CpUB2GM<uint64_t>(seqOutSwapIn, ub_buff, sizeof(uint64_t));
        }
    }

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
            const int flagNum = MAX_BLOCK_NUM * 2;
            for (int i = 0; i < flagNum * FLAG_UNIT_INT_NUM; ++i) {
                *(ub_buff + i) = 0;
            }
            CpUB2GM<uint8_t>(swapFlagSwapIn, (__ubuf__ uint8_t *)ub_buff, flagNum * FLAG_UNIT_INT_NUM * sizeof(uint64_t));
            CpUB2GM<uint8_t>(swapFlagSwapOut, (__ubuf__ uint8_t *)ub_buff, flagNum * FLAG_UNIT_INT_NUM * sizeof(uint64_t));
        }
    }

    __aicore__ inline void SyncPreprocess()
    {
        __ubuf__ uint64_t *ub_buff = (__ubuf__ uint64_t *)get_imm(0);
        __gm__ uint64_t *syncAllFlag = (__gm__ uint64_t *)swapFlagSwapIn + MAX_BLOCK_NUM * FLAG_UNIT_INT_NUM;
        if (blockIdx == 0) {
            ClearFlag();
            SetFlag(ub_buff, syncAllFlag, 10086);
        } else {
            CheckFlag(ub_buff, syncAllFlag, 10086);
        }
    }

    __aicore__ inline void SyncPostprocess()
    {
        __ubuf__ uint64_t *ub_buff = (__ubuf__ uint64_t *)get_imm(0);
        __gm__ uint64_t *syncAllFlag = (__gm__ uint64_t *)swapFlagSwapOut + MAX_BLOCK_NUM * FLAG_UNIT_INT_NUM;
        if (blockIdx != 0) {
            SetFlag(ub_buff, syncAllFlag + blockIdx * FLAG_UNIT_INT_NUM, 10087);
        } else {
            for (int i = 1; i < blockNum; ++i) {
                CheckFlag(ub_buff, syncAllFlag + i * FLAG_UNIT_INT_NUM, 10087);
            }
            ClearFlag();
        }
    }

private:
    RmaShmHeader queueHeader;
    RmaShmDataHead dataHeadSwapOut;
    GM_ADDR updateTables[MAX_TABLE_NUM];
    uint64_t tableNum;
    GM_ADDR swapInIndex;
    GM_ADDR swapOutIndex;
    uint64_t swapInLen;     // 换入emb条数
    uint64_t swapOutLen;    // 换出emb条数
    GM_ADDR svmBuffSwapIn;
    GM_ADDR svmBuffSwapOut;
    GM_ADDR usrWorkspace;
    GM_ADDR swapFlagSwapIn;
    GM_ADDR swapFlagSwapOut;
    uint64_t embDim;        // 每条emb的维度，单位字节
    uint64_t embDimSplit;   // 分表后每条emb的维度，单位字节
    GM_ADDR embSwapCache;
    uint64_t cacheCapacity; // 数据缓存能容纳的emb条数
    uint64_t cacheFront;    // 数据缓存队列头索引
    uint64_t cacheRear;     // 数据缓存队列尾索引
    GM_ADDR output;         // 算子返回值
    uint32_t processBlockNum;
    uint32_t processBlockIdx;
};

#endif //LCCL_RMA_SWAP_MULTI_TABLES_H