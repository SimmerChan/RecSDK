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

#ifndef LCCL_ALL2ALLVC_H
#define LCCL_ALL2ALLVC_H

#include "collectives.h"
#include "ipc_queue.h"

using namespace AscendC;

template <typename T>
class All2AllVC : public Collectives {
    constexpr static int INVALID_RANK_NUM = 0xFFFFFFFF;  // 非法rank
    constexpr static int64_t CORE_NUMS_PER_STAGE = 16;   // 每个阶段提供的最大核数
    constexpr static int64_t SHARE_QUE_DEPTH = 8;        // 单个共享队列深度
    constexpr static int64_t DATA_SLICE = 2;
    constexpr static int64_t STAGE_NUM = 2;
    constexpr static int64_t SINGLE_RANK_MAX_NUM = CORE_NUMS_PER_STAGE;
    constexpr static int64_t MULTI_RANK_SIZE = (LCAL_MAX_RANK_SIZE + SINGLE_RANK_MAX_NUM - 1) / SINGLE_RANK_MAX_NUM;

    constexpr static int64_t IDLER_CORE = 0;        // 闲置的核
    constexpr static int64_t PRODUCER_CORE_X = 1;   // 生产组，负责向共享内存写入数据，input->ipc
    constexpr static int64_t PRODUCER_CORE_Y = 2;   // 生产组，负责向共享内存写入数据，input->ipc
    constexpr static int64_t CONSUMER_CORE_X = 3;   // 消费组X
    constexpr static int64_t CONSUMER_CORE_Y = 4;   // 消费组Y

public:
    __aicore__ inline All2AllVC(int rank, int rankSize, uint32_t extraFlag)
        : Collectives(rank, rankSize, extraFlag)
    {
    }

    __aicore__ inline void Init(KERNELS_ARGS_CF_FUN())
    {
        Collectives::Init(KERNELS_ARGS_CALL());
        this->localRankSize = localRankSize;
        this->xRankSize = localRankSize;
        this->yRankSize = rankSize / localRankSize;
        this->xRankIdx = rank % localRankSize;
        this->yRankIdx = rank / localRankSize;

        blockIdx = GetBlockIdx();
        blockNum = GetBlockNum();

        sendCountMatrixGm.SetGlobalBuffer((__gm__ int64_t *)send_count_matrix, rankSize * rankSize);
        // 初始化共享内存信息
        InitShare();
        // 初始化核分组
        InitCoreGroup();
        // 初始化数据切片
        InitDataSlice();

        // 初始化输入输出
        inputGt.SetGlobalBuffer((__gm__ T *)input);
        outputGt.SetGlobalBuffer((__gm__ T *)output);
    }

    __aicore__ inline void Process()
    {
        if (coreGroup == PRODUCER_CORE_X || coreGroup == PRODUCER_CORE_Y) {
            ProducerStage();
        } else if (coreGroup == CONSUMER_CORE_X || coreGroup == CONSUMER_CORE_Y) {
            ConsumerStage();
        }
    }

private:
    // 计算rank数量较大时的queNum 以及  每个队列里单块可放入的元素数量queElemLen
    __aicore__ inline void InitShare()
    {
        int64_t queNum = xRankSize + yRankSize;
        int64_t queElemNum = xRankSize * yRankSize * 2;
        queElemLen = IPC_BUFF_MAX_SIZE / sizeof(T) / queElemNum / SHARE_QUE_DEPTH;  // 计算共享队列元素大小
    }

    __aicore__ inline void InitCoreGroup()
    {
        coreNumPerRank = 1;
        rankNumPerCore = 1;
        // 多卡下为coreNumsPerStage
        coreNumPerStage = xRankSize + yRankSize;
        if (blockIdx < STAGE_NUM * coreNumPerStage) {
            if (blockIdx < coreNumPerStage) {
                coreGroup = blockIdx % coreNumPerStage < xRankSize ? PRODUCER_CORE_X : PRODUCER_CORE_Y;
                targetRank[0] = (coreGroup == PRODUCER_CORE_X) ? yRankIdx * xRankSize + blockIdx
                                                               : (blockIdx - xRankSize) * xRankSize + xRankIdx;
            } else {
                coreGroup = blockIdx % coreNumPerStage < xRankSize ? CONSUMER_CORE_X : CONSUMER_CORE_Y;
                targetRank[0] = (coreGroup == CONSUMER_CORE_X)
                                    ? yRankIdx * xRankSize + blockIdx % coreNumPerStage
                                    : (blockIdx % coreNumPerStage - xRankSize) * xRankSize + xRankIdx;
            }
            targetXRankIdx = targetRank[0] % xRankSize;
            targetYRankIdx = targetRank[0] / xRankSize;
        } else {
            coreGroup = IDLER_CORE;
        }
    }

    __aicore__ inline void InitDataSlice()
    {
        queLen = IPC_BUFF_MAX_SIZE / sizeof(T) / (xRankSize * yRankSize * STAGE_NUM);  // 一个que的可放入的元素数量
        queSize = queLen * sizeof(T);

        // 生产者负责搬运本rank的输入数据至共享内存，input-->share
        if (coreGroup == PRODUCER_CORE_X || coreGroup == PRODUCER_CORE_Y) {
            ProducerDataSlice();
        } else if (coreGroup == CONSUMER_CORE_X || coreGroup == CONSUMER_CORE_Y) {
            ConsumerDataSlice();
        }
    }

    __aicore__ inline void ProducerDataSlice()
    {
        maxSliceNum = 0;
        if (coreGroup == PRODUCER_CORE_X) {
            for (int i = 0; i < yRankSize; i++) {
                writeQue[i].Init(&sync,
                    magic,
                    shareAddrs[targetRank[0]] + IPC_DATA_OFFSET + (xRankIdx * yRankSize + i) * queSize,
                    queLen,
                    queElemLen);
                sendOffset[i] = 0;
                for (int j = 0; j < (targetXRankIdx + i * xRankSize); j++) {
                    sendOffset[i] += sendCountMatrixGm.GetValue(rank * rankSize + j);
                }
                inputDataLen[i] =
                    sendCountMatrixGm.GetValue(rank * rankSize + (targetXRankIdx + i * xRankSize)) / DATA_SLICE;
                SplitData(inputDataLen[i], 1, 0, inputOffset[i], inputLen[i], sendOffset[i]);
                // 当前核负责的数据切片数，能分成一个que中的多少小块
                sliceNum[i] = CeilDiv(inputLen[i], queElemLen);
                if (sliceNum[i] > maxSliceNum) {
                    maxSliceNum = sliceNum[i];
                }
            }
        } else {
            for (int i = 0; i < xRankSize; i++) {
                writeQue[i].Init(&sync,
                    magic,
                    shareAddrs[targetRank[0]] + IPC_DATA_OFFSET +
                        (xRankSize * yRankSize + yRankIdx * xRankSize + i) * queSize,
                    queLen,
                    queElemLen);
                sendOffset[i] = 0;
                for (int j = 0; j < (targetYRankIdx * xRankSize + i); j++) {
                    sendOffset[i] += sendCountMatrixGm.GetValue(rank * rankSize + j);
                }
                int inputDataLenTmp =
                    sendCountMatrixGm.GetValue(rank * rankSize + (targetYRankIdx * xRankSize + i)) / DATA_SLICE;
                sendOffset[i] += inputDataLenTmp;
                inputDataLen[i] =
                    sendCountMatrixGm.GetValue(rank * rankSize + (targetYRankIdx * xRankSize + i)) - inputDataLenTmp;
                SplitData(inputDataLen[i], 1, 0, inputOffset[i], inputLen[i], sendOffset[i]);
                // 当前核负责的数据切片数，能分成一个que中的多少小块
                sliceNum[i] = CeilDiv(inputLen[i], queElemLen);
                if (sliceNum[i] > maxSliceNum) {
                    maxSliceNum = sliceNum[i];
                }
            }
        }
    }

    __aicore__ inline void ConsumerDataSlice()
    {
        maxSliceNum = 0;
        if (coreGroup == CONSUMER_CORE_X) {
            for (int k = 0; k < yRankSize; k++) {
                readQue[k].Init(&sync,
                    magic,
                    shareAddrs[targetRank[0]] + IPC_DATA_OFFSET +
                        (xRankSize * yRankSize + k * xRankSize + xRankIdx) * queSize,
                    queLen,
                    queElemLen);
                revOffset[k] = 0;
                for (int j = 0; j < (k * xRankSize + targetXRankIdx); j++) {
                    revOffset[k] += sendCountMatrixGm.GetValue(j * rankSize + rank);
                }
                int inputDataLenTmp =
                    sendCountMatrixGm.GetValue((k * xRankSize + targetXRankIdx) * rankSize + rank) / DATA_SLICE;
                revOffset[k] += inputDataLenTmp;
                outputDataLen[k] =
                    sendCountMatrixGm.GetValue((k * xRankSize + targetXRankIdx) * rankSize + rank) - inputDataLenTmp;
                SplitData(outputDataLen[k], 1, 0, outputOffset[k], outputLen[k], revOffset[k]);
                // 当前核负责的数据切片数，能分成一个que中的多少小块
                sliceNum[k] = CeilDiv(outputLen[k], queElemLen);
                if (sliceNum[k] > maxSliceNum) {
                    maxSliceNum = sliceNum[k];
                }
            }
        } else {
            for (int k = 0; k < xRankSize; k++) {
                readQue[k].Init(&sync,
                    magic,
                    shareAddrs[targetRank[0]] + IPC_DATA_OFFSET + (k * yRankSize + yRankIdx) * queSize,
                    queLen,
                    queElemLen);
                revOffset[k] = 0;
                for (int j = 0; j < (xRankSize * targetYRankIdx + k); j++) {
                    revOffset[k] += sendCountMatrixGm.GetValue(j * rankSize + rank);
                }
                outputDataLen[k] =
                    sendCountMatrixGm.GetValue((xRankSize * targetYRankIdx + k) * rankSize + rank) / DATA_SLICE;
                SplitData(outputDataLen[k], 1, 0, outputOffset[k], outputLen[k], revOffset[k]);
                // 当前核负责的数据切片数，能分成一个que中的多少小块
                sliceNum[k] = CeilDiv(outputLen[k], queElemLen);
                if (sliceNum[k] > maxSliceNum) {
                    maxSliceNum = sliceNum[k];
                }
            }
        }
    }

    __aicore__ inline void SplitData(const int64_t totalLen, const int64_t useCoreNum, const int64_t useCoreIdx,
                                    int64_t &dataOffset, int64_t &dataLen, int startOffset)
    {
        // 向上整除获取每个core切分的数据个数
        dataLen = CeilDiv(totalLen, useCoreNum);
        // 数据量极小或略微超过核数的情况，后面若干个core数据量为0
        dataOffset = useCoreIdx * dataLen + startOffset;  // 使用当前block在useBlock里的相对索引来计算偏移
        if (useCoreIdx * dataLen >= totalLen) {
            dataOffset = totalLen + startOffset;
            dataLen = 0;
            return;
        }
        // 非整除情况，最后一个core数据量为剩余数据量
        if (dataOffset + dataLen - startOffset > totalLen) {
            dataLen = totalLen - useCoreIdx * dataLen;
        }
    }

    __aicore__ inline void ProducerStage()
    {
        for (auto i = 0; i < rankNumPerCore; ++i) {
            if (targetRank[i] == INVALID_RANK_NUM) {
                continue;
            }
            // 写共享内存队列时，需要等待当前rank
            waitRankListForWrite[i][0] = targetRank[i];
            waitNumForWrite[i] = 1;
        }
        InputToSharePipeline();
    }

    __aicore__ inline void InputToSharePipeline()
    {
        int64_t flagValue[MULTI_RANK_SIZE];  // 要等待标志位的储存值
        for (auto i = 0; i < rankNumPerCore; ++i) {
            flagValue[i] = -1;  // 统一赋值为-1，便于后续小于判断
        }
        // 以最多切片sliceNum[0]为切片数进行循环，切片数不足的不拷贝
        for (auto sliceIdx = 0; sliceIdx < maxSliceNum; ++sliceIdx) {
            for (auto i = 0; i < rankNumPerCore; ++i) {
                if (targetRank[i] == INVALID_RANK_NUM) {
                    continue;
                }
                InputToShareSlice(i, sliceIdx, flagValue[i]);
            }
        }
    }

    __aicore__ inline void InputToShareSlice(int64_t idx, int64_t sliceIdx, int64_t &flagValue)
    {
        // 计算当前切片拷贝数据量，数据量为0时不拷贝
        int tmpRankSize = (coreGroup == PRODUCER_CORE_X) ? yRankSize : xRankSize;
        for (int i = 0; i < tmpRankSize; i++) {
            copyLen = inputLen[i] - queElemLen * sliceIdx;
            if (copyLen > queElemLen) {
                copyLen = queElemLen;
            } else if (copyLen <= 0) {
                copyLen = 0;
                continue;
            }
            readGt = inputGt[sliceIdx * queElemLen + inputOffset[i]];
            // 这里一共定义了xranksize*yranksize *2 个标志位，两个方向各使用一半，前一半以[x,y]为shape考虑
            // [j,k]表示x方向第j个队列发给y方向第k个rank的；y方向反之
            int tmpRankIdx = (coreGroup == PRODUCER_CORE_X) ? xRankIdx * yRankSize + i
                                                            : xRankSize * yRankSize + yRankIdx * xRankSize + i;
            writeQue[i].DeQue(waitRankListForWrite[idx], 1, tmpRankIdx + xRankSize * yRankSize * STAGE_NUM);
            writeGt = writeQue[i].EnQue();
            if (copyLen > 0) {
                CpGM2GMPingPong<T>(copyLen * sizeof(T), readGt, writeGt, COPYONLY);
            }
            sync.SetInnerFlag(magic, sliceIdx, targetRank[idx], tmpRankIdx);
        }
    }

    __aicore__ inline void ConsumerStage()
    {
        int64_t flagValue[MULTI_RANK_SIZE];  // 要等待标志位的储存值
        int tmpRankSize = (coreGroup == CONSUMER_CORE_X) ? yRankSize : xRankSize;
        for (auto i = 0; i < tmpRankSize; ++i) {
            flagValue[i] = -1;
        }
        // 以最多切片sliceNum[0]为切片数进行循环，切片数不足的不拷贝
        for (auto sliceIdx = 0; sliceIdx < maxSliceNum; ++sliceIdx) {
            for (auto i = 0; i < rankNumPerCore; ++i) {
                if (targetRank[i] == INVALID_RANK_NUM) {
                    continue;
                }
                ShareToOutputSlice(i, sliceIdx, flagValue);
            }
        }
    }

    __aicore__ inline void ShareToOutputSlice(int64_t idx, int64_t sliceIdx, int64_t *flagValue)
    {
        int tmpRankSize = (coreGroup == CONSUMER_CORE_X) ? yRankSize : xRankSize;
        for (int i = 0; i < tmpRankSize; i++) {
            copyLen = outputLen[i] - queElemLen * sliceIdx;
            if (copyLen > queElemLen) {
                copyLen = queElemLen;
            } else if (copyLen <= 0) {
                copyLen = 0;
                continue;
            }
            // consume的x方向，查看target上的flag后半部分[ysize,xsize]，来判断上一步y方向上的第tmpRankSize个rank是否发给target了
            int tmpRankIdx = (coreGroup == CONSUMER_CORE_X) ? xRankSize * yRankSize + i * xRankSize + xRankIdx
                                                            : i * yRankSize + yRankIdx;
            if (flagValue[i] < sliceIdx) {
                sync.WaitInnerFlag(magic, sliceIdx, targetRank[idx], tmpRankIdx);
                flagValue[i] = sync.GetInnerFlag(targetRank[idx], tmpRankIdx) & EVENT_ID_MASK;
            }
            readGt = readQue[i].ReadFront();
            writeGt = outputGt[sliceIdx * queElemLen + outputOffset[i]];
            if (copyLen > 0) {
                CpGM2GMPingPong<T>(copyLen * sizeof(T), readGt, writeGt, COPYONLY);
            }
            sync.SetInnerFlag(magic, sliceIdx, targetRank[idx], tmpRankIdx + xRankSize * yRankSize * STAGE_NUM);
        }
        if (sliceIdx == maxSliceNum - 1) {
            for (int i = 0; i < tmpRankSize; i++) {
                int tmpRankIdx = (coreGroup == CONSUMER_CORE_X) ? xRankSize * yRankSize + i * xRankSize + xRankIdx
                                                                : i * yRankSize + yRankIdx;
                sync.SetInnerFlag(0, 0, targetRank[idx], tmpRankIdx + xRankSize * yRankSize * STAGE_NUM);
                sync.SetInnerFlag(0, 0, targetRank[idx], tmpRankIdx);
            }
        }
    }

    GlobalTensor<T> inputGt;
    GlobalTensor<T> outputGt;
    GlobalTensor<T> readGt;
    GlobalTensor<T> writeGt;
    GlobalTensor<int64_t> sendCountMatrixGm;
    int targetXRankIdx;
    int targetYRankIdx;
    int coreNumsPerStage;
    int64_t maxSliceNum;
    int64_t revLen = 0;
    int64_t sendLen = 0;
    int64_t sendOffset[MULTI_RANK_SIZE];
    int64_t revOffset[MULTI_RANK_SIZE];
    int64_t inputDataLen[MULTI_RANK_SIZE];
    int64_t outputDataLen[MULTI_RANK_SIZE];

    int waitRankListForWrite[MULTI_RANK_SIZE][1];  // 写共享内存时，需要等待的rank列表
    int waitNumForWrite[MULTI_RANK_SIZE];          // 写共享内存时，需要等待的数量
    int waitBlockForWrite[MULTI_RANK_SIZE];        // 写共享内存时，需要等待的标志位

    int64_t queLen;
    int64_t queSize;
    int64_t coreNumPerStage;                // 每个阶段使用的核数
    int64_t coreNumPerRank;                 // 每个rank数据分配的核数
    int64_t rankNumPerCore;                 // 每个核负责的rank数
    int64_t coreGroup;                      // 当前核的功能分组
    int64_t groupCoreIdx[MULTI_RANK_SIZE];  // 当前核在组内的索引，可以为等效核索引
    int64_t targetRank[MULTI_RANK_SIZE];    // 当前核负责的rank

    IpcQueue<T> readQue[MULTI_RANK_SIZE];   // 读端共享内存队列
    IpcQueue<T> writeQue[MULTI_RANK_SIZE];  // 写端共享内存队列
    int64_t queElemLen;                     // 共享内存队列里每个元素大小（以T计）

    int64_t sliceNum[MULTI_RANK_SIZE];      // 当前核负责的数据切片总数
    int64_t copyLen;                        // 当前拷贝数据片的长度（以T计）
    int64_t inputOffset[MULTI_RANK_SIZE];   // 当前核负责的input偏移（以T计）
    int64_t inputLen[MULTI_RANK_SIZE];      // 当前核负责的input长度（以T计）
    int64_t outputOffset[MULTI_RANK_SIZE];  // 当前核负责的output偏移（以T计）
    int64_t outputLen[MULTI_RANK_SIZE];     // 当前核负责的output长度（以T计）
};

#endif // LCCL_ALL2ALLVC_H
