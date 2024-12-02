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
#ifndef RMA_SHM_SVM_H
#define RMA_SHM_SVM_H

constexpr int32_t RMA_SHM_HEAD_LEN = 128; /* SHM的队列头长度，用于保存全局信息 */
constexpr int32_t RMA_SHM_DATA_HEAD = 56; /* 数据元素的头长度 */
constexpr int32_t RMA_SHM_READY_LEN = 48; /* 数据元素的头的readyLen偏移 */
constexpr int32_t RMA_DIM_MAX = 2;

enum class RmaDevModel {
    MEM_MAP_DEV,    // 910_93需要更新驱动支持
    SVM_MAP_DEV,    // 910_93
    PCIE_TH_DEV     // 910B
};

// 队列头定义
struct RmaShmHeader {
    uint64_t queueCapacity; /* 队列深度 */
    uint64_t totalMemSize;  /* 总内存大小 */
    uint64_t seqIn;         /* 最新写入的seq */
    uint64_t seqOut;        /* 最新读取的seq */
    uint64_t frontOffset;   /* 可访问的内存地址偏移量 */
    uint64_t tailOffset;    /* 可写入数据的地址偏移量，从队列尾部写入 */
    uint64_t buffLimit;     /* 队列尾部无法写入数据的偏移，标识队列需要返回到头部进行写入 */
};

// 队列中每个元素的头定义
struct RmaShmData {
    uint64_t totalLen;         /* 每个元素的总长度，单位byte */
    uint64_t sequence;         /* 元素序列号 */
    int32_t dataType;          /* 数据类型 */
    int32_t dimNum;            /* dim的维度数目 */
    int64_t dims[RMA_DIM_MAX]; /* shape值 */
    uint64_t dataLen;          /* 数据长度，单位byte */
    uint64_t readyLen;         /* 已准备好的数据长度，单位byte */
};

int64_t GetShmAddr(std::string name, int rankId, int capacity);
void *GetHostAddr(std::string name);
void FreeShmAddr(int deviceId);
uint8_t *ShmOutqueue(RmaShmHeader *header);
int64_t GetShmElemNum(RmaShmHeader *header);
void SetShmQueueSeqOut(RmaShmHeader *header, uint64_t sequence);
uint64_t GetShmSeq(RmaShmHeader *queueHeader);
void ClearShmQueue();
uint8_t *ShmEnqueueHeadRaw(RmaShmHeader *header, int64_t dims[], uint64_t sequence);
uint8_t *ShmEnqueueGetLast(RmaShmHeader *header, int64_t dims[]);

#endif  // RMA_SHM_SVM_H
