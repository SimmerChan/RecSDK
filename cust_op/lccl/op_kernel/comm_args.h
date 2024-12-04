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

#ifndef LCCL_COMM_ARGS_H
#define LCCL_COMM_ARGS_H

#include "kernel_operator.h"
#include <limits.h>

constexpr int32_t MAX_BLOCK_NUM = 48;
constexpr int64_t FLAG_UNIT_INT_NUM = 4;    // 同步标志位占用长度 4 * 8 B
constexpr int64_t TIME_OUT = 375000000;     // 超时等待时间   大概5分钟

constexpr int32_t RMA_UB_B4_BUFF_OFFSET = 64;
constexpr int32_t RMA_UB_B8_BUFF_OFFSET = 128;
constexpr int32_t RMA_UB_DATA_BUFF_OFFSET = 256;
constexpr int32_t RMA_SHM_HEAD_LEN = 128;      // 队列头长度
constexpr int32_t RMA_SHM_DATA_HEAD = 56;      // 数据头长度
constexpr int32_t UNIT_COPY_SIZE = 190 * 1024; // UB复制大小
constexpr int32_t RMA_SHAPE_DIM_MAX = 2;

constexpr uint64_t RMA_WORK_SPACE_SIZE = 202 * 1024 * 1024;
constexpr uint64_t SWAP_CACHE_SIZE = 100 * 1024 * 1024;  // 换入/换出数据缓存区大小为10MB
constexpr uint64_t SWAP_IN_FLAG_OFFSET = 0;             // 换入标志位偏移
constexpr uint64_t SWAP_IN_CACHE_OFFSET = 1 * 1024 * 1024;  // 换入数据缓存区偏移
constexpr uint64_t SWAP_OUT_FLAG_OFFSET = 101 * 1024 * 1024; // 换出标志位偏移
constexpr uint64_t SWAP_OUT_CACHE_OFFSET = 102 * 1024 * 1024;// 换出数据缓存区偏移

// 队列头定义
struct RmaShmHeader {
    uint64_t queueCapacity;     // 队列容量/深度
    uint64_t totalMemSize;      // 总内存占用
    uint64_t seqIn;             // 入队序列号
    uint64_t seqOut;            // 出队序列号
    uint64_t frontOffset;       // 队列头元素偏移
    uint64_t tailOffset;        // 队列尾元素偏移
    uint64_t buffLimit;
    uint64_t seqOutPre;
    uint64_t frontOffsetPre;
};

// 队列头各参数偏移，8字节为单位
struct RmaQueueOffset {
    static constexpr int32_t RMA_CAPACITY_OFFSET = 0;
    static constexpr int32_t RMA_TOTAL_SIZE_OFFSET = 1;
    static constexpr int32_t RMA_SEQ_IN_OFFSET = 2;
    static constexpr int32_t RMA_SEQ_OUT_OFFSET = 3;
    static constexpr int32_t RMA_QUEUE_FRONT_OFFSET = 4;
    static constexpr int32_t RMA_QUEUE_TAIL_OFFSET = 5;
    static constexpr int32_t RMA_BUFF_LIMIT_OFFSET = 6;
    static constexpr int32_t RMA_SEQ_OUT_PRE_OFFSET = 7;
    static constexpr int32_t RMA_QUEUE_FRONT_PRE_OFFSET = 8;
};

// 队列中每个元素的头定义
struct RmaShmDataHead {
    uint64_t totalLen;                  /* 每个元素的总长度，单位byte */
    uint64_t sequence;                  /* 元素序列号 */
    int32_t dataType;                   /* 数据类型 */
    int32_t dimNum;                     /* dim的维度数目 */
    int64_t dims[RMA_SHAPE_DIM_MAX];    /* shape值 */
    uint64_t dataLen;                   /* 数据长度，单位byte */
    uint64_t readyLen;                  /* 已准备好的数据长度，单位byte */
};
constexpr int32_t RMA_READY_LEN_OFFSET = 6;

#endif //LCCL_COMM_ARGS_H
