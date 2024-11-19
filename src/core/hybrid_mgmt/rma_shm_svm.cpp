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

#include <acl/acl.h>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <string>
#include <stdio.h>
#include "securec.h"
#include <sys/stat.h>
#include <sys/shm.h>
#include <driver/ascend_hal_define.h>
#include <unordered_map>
#include "rma_shm_svm.h"
#include "utils/common.h"

using namespace std;

extern "C" {
drvError_t halHostRegister(void *srcPtr, UINT64 size, UINT32 flag, UINT32 devid, void **dstPtr);
drvError_t halHostUnregister(void *srcPtr, UINT32 devid);
drvError_t rtDeviceGetBareTgid(uint32_t *pid);
}

constexpr int32_t HUGEPAGE_ENABLE = 1;
const uint64_t RMA_SHM_TOTAL_MEM_SIZE = 1 * 1024 * 1024 * 1024 * 1L; // 内存总容量
constexpr int RMA_SHM_QUEUE_CAPACITY = 50;                           // 队列的最大深度
constexpr int32_t RANK_SIZE = 16;

uint32_t g_pid = 0;
bool g_aclInit[RANK_SIZE] = {false};
std::unordered_map<std::string, void *> g_shmSvmMap;
std::unordered_map<std::string, void *> g_shmAddr;
std::unordered_map<std::string, int> g_shmId;

typedef enum tagRmaDevModel {
    MEM_MAP_DEV,
    SVM_MAP_DEV,
    PCIE_TH_DEV
} RmaDevModel_t;

int32_t g_rmaDevModel = SVM_MAP_DEV; // 910C

void InitShmHeader(void *shmHeader, int64_t memSize, int32_t capacity)
{
    RmaShmHeader *header = (RmaShmHeader *)shmHeader;
    header->totalMemSize = memSize - RMA_SHM_HEAD_LEN;
    header->queueCapacity = capacity;
    header->seqIn = 0;
    header->seqOut = 0;
    header->frontOffset = RMA_SHM_HEAD_LEN;
    header->tailOffset = RMA_SHM_HEAD_LEN;
    header->buffLimit = 0;
}

void ResetShmHeader(void *shmHeader)
{
    RmaShmHeader *header = (RmaShmHeader *)shmHeader;
    header->seqIn = 0;
    header->seqOut = 0;
    header->frontOffset = RMA_SHM_HEAD_LEN;
    header->tailOffset = RMA_SHM_HEAD_LEN;
    header->buffLimit = 0;
}

void RmaFreeShm(std::string shmName, void *memory)
{
    if (g_rmaDevModel == SVM_MAP_DEV) {
        if (aclrtFreeHost(memory) != ACL_ERROR_NONE) {
            LOG_ERROR("free host mem failed.");
        }
    } else {
        int shmId = g_shmId[shmName];
        LOG_INFO("free shm shmid: {}", shmId);
        (void)shmdt(memory);

        shmctl(shmId, IPC_RMID, nullptr);
        g_shmId.erase(shmName);
    }
}

// aicore申请shm内存
void *RmaCreateShm(std::string shmName, uint64_t memSize, int deviceId, int capacity)
{
    void *memory = nullptr;
    if (g_rmaDevModel == SVM_MAP_DEV) {
        if (aclrtMallocHost((void **)&memory, memSize) != ACL_ERROR_NONE) {
            LOG_ERROR("Malloc host memory failed");
            return nullptr;
        }
        (void)aclrtMemset(memory, memSize, 0, memSize);
        LOG_INFO("create memory {}, size: {} bytes", shmName.c_str(), memSize);
    } else {
        struct shmid_ds buf;
        key_t key = static_cast<key_t>(std::hash<std::string> {}(shmName));
#if HUGEPAGE_ENABLE
        int shmId = shmget(key, memSize, IPC_CREAT | 0666 | SHM_HUGETLB);
#else
        int shmId = shmget(key, memSize, IPC_CREAT | 0600); // 0600提供文件所有者有读和写的权限
#endif

        LOG_INFO("create shm {}, shmid: {}, size: {} bytes", shmName.c_str(), shmId, memSize);
        if (shmId == -1) {
            LOG_ERROR("shmget failed");
            return nullptr;
        }

        void *memory = nullptr;
        memory = shmat(shmId, nullptr, 0);
        if (memory == reinterpret_cast<void *>(-1)) {
            LOG_ERROR("shmat failed");
            shmctl(shmId, IPC_RMID, nullptr);
            return nullptr;
        }

        shmctl(shmId, IPC_STAT, &buf);
        (void)memset(memory, 0, memSize);
        g_shmId.insert(std::make_pair(shmName, shmId));
    }

    uint32_t flag;
    switch (g_rmaDevModel) {
        case MEM_MAP_DEV:
            flag = HOST_MEM_MAP_DEV;
            break;
        case SVM_MAP_DEV:
            flag = HOST_SVM_MAP_DEV;
            break;
        default :
            flag = HOST_MEM_MAP_DEV_PCIE_TH;
            break;
    }

    void *svmMem = nullptr;
    if (halHostRegister(memory, memSize, flag, deviceId, &svmMem) != DRV_ERROR_NONE) {
        LOG_ERROR("rank {} halHostRegister failed", deviceId);
        RmaFreeShm(shmName, memory);
        return nullptr;
    }

    g_shmAddr.insert(std::make_pair(shmName, memory));

    // 初始化队列头
    InitShmHeader(memory, memSize, capacity);

    return svmMem;
}

// 仅用于pybind侧调用，创建共享内存
int64_t GetShmAddr(std::string name, int rankId, int capacity)
{
    LOG_INFO("rank {}, alloc shm {}", rankId, name.c_str());

    if (rankId >= RANK_SIZE) {
        LOG_ERROR("rank {} is invalid", rankId);
        return reinterpret_cast<int64_t>(nullptr);
    }

    auto memSize = RMA_SHM_TOTAL_MEM_SIZE;
    if (capacity > RMA_SHM_QUEUE_CAPACITY) {
        capacity = RMA_SHM_QUEUE_CAPACITY;
    }

    if (!g_aclInit[rankId]) {
        aclInit(nullptr);
        aclrtSetDevice(rankId);
        g_aclInit[rankId] = true;
    }

    if (g_pid == 0) {
        (void)rtDeviceGetBareTgid(&g_pid);
    }
    std::string shmName = name + "_" + std::to_string(g_pid);
    LOG_INFO("rank {}, alloc shm-name {}", rankId, shmName.c_str());

    if (g_shmSvmMap.find(shmName) == g_shmSvmMap.end()) {
        g_shmSvmMap.insert(std::make_pair(shmName, RmaCreateShm(shmName, memSize, rankId, capacity)));
    }

    return reinterpret_cast<int64_t>(g_shmSvmMap[shmName]);
}

// 仅用于pybind侧调用，获取host侧地址；
int64_t GetShmHost(std::string name, int rankId)
{
    std::string shmName = name + "_" + std::to_string(g_pid);
    LOG_INFO("rank {}, get shm-name {}", rankId, shmName.c_str());

    auto *hostAddr = g_shmAddr[shmName];
    if (hostAddr == nullptr) {
        LOG_ERROR("shm add is invalid");
        return reinterpret_cast<int64_t>(nullptr);
    }

    return reinterpret_cast<int64_t>(hostAddr);
}

// 仅用于hd_transfer的send/recv调用，获取host侧地址；
void *GetHostAddr(std::string name, int rankId)
{
    std::string shmName = name + "_" + std::to_string(g_pid);

    return g_shmAddr[shmName];
}

void FreeShmAddr(int deviceId)
{
    if (!g_aclInit[deviceId]) {
        return;
    }

    for (auto &pair : g_shmAddr) {
        halHostUnregister(pair.second, deviceId);
        RmaFreeShm(pair.first, pair.second);
        pair.second = nullptr;
        LOG_INFO("rank {} free memory: {} success", deviceId, pair.first.c_str());
    }
    g_shmAddr.clear();
    g_shmId.clear();
    g_shmSvmMap.clear();

    if (g_aclInit[deviceId]) {
        aclrtResetDevice(deviceId);
        aclFinalize();
        g_aclInit[deviceId] = false;
        LOG_INFO("rank {} free acl device", deviceId);
    }
}

uint64_t GetShmSeq(RmaShmHeader *queueHeader)
{
    uint64_t sequence = queueHeader->seqIn;
    return ++sequence;
}

void ClearShmQueue()
{
    for (auto &pair : g_shmAddr) {
        // 复位队列头
        ResetShmHeader(pair.second);
        LOG_INFO("reset queue: {}", pair.first);
    }
}

/**
 * @brief 返回数据元素的数据头地址
 * @param header
 * @param memData
 * @param dims
 * @param sequence
 * @return
 */
uint8_t *ShmEnqueueRaw(RmaShmHeader *header, const void *memData, int64_t dims[RMA_DIM_MAX], uint64_t sequence)
{
    int64_t dataSize = dims[0] * dims[1] * sizeof(float) * 1L;
    uint8_t *lastPos = nullptr;
    RmaShmData dataHead;
    uint64_t *out = (uint64_t *)(&header->buffLimit) + 1;

    LOG_INFO("before enqueue, capacity: {}, seq-in: {}, seq-out: {}, out:[ {} {} ]", header->queueCapacity,
             header->seqIn, header->seqOut, *out, *(out + 1));
    LOG_INFO("head: {}, tail: {}, buff-limit: {}", header->frontOffset, header->tailOffset, header->buffLimit);

    int64_t queueNum = header->seqIn - header->seqOut;
    if (queueNum >= header->queueCapacity) {
        LOG_ERROR("rma queue is full, num: {}", queueNum);
        return nullptr;
    }

    dataHead.totalLen = dataSize + RMA_SHM_DATA_HEAD;
    dataHead.dataType = 0;
    dataHead.dimNum = RMA_DIM_MAX;
    dataHead.dataLen = dataSize;
    dataHead.sequence = sequence;
    dataHead.dims[0] = dims[0];
    dataHead.dims[1] = dims[1];
    dataHead.readyLen = dataHead.dataLen;

    if (header->tailOffset + dataSize > header->totalMemSize) {
        // 如果队列尾部的空余放不下新插入的数据，则从队列头部插入，当前约束队列不够大，不会被写满
        lastPos = reinterpret_cast<uint8_t *>(header) + RMA_SHM_HEAD_LEN;
        if (memcpy_s(lastPos, RMA_SHM_DATA_HEAD, &dataHead, RMA_SHM_DATA_HEAD) != EOK) {
            LOG_ERROR("memcpy failed");
            return nullptr;
        }

        header->buffLimit = header->tailOffset; // 标识该位置后面无可读取的数据，需要返回到队列首部
        header->tailOffset = RMA_SHM_HEAD_LEN + dataHead.totalLen; // 从队列首部开始偏移
    } else {
        // 正常顺序入队列
        lastPos = reinterpret_cast<uint8_t *>(header) + header->tailOffset;
        if (memcpy_s(lastPos, RMA_SHM_DATA_HEAD, &dataHead, RMA_SHM_DATA_HEAD) != EOK) {
            LOG_ERROR("memcpy failed");
            return nullptr;
        }

        header->tailOffset += dataHead.totalLen; // 尾部往后偏移，指导下一个元素的插入位置
    }

    header->seqIn = sequence; // 更新队列头的Seq

    LOG_INFO("after enqueue, capacity: {}, seq-in: {}, seq-out: {}", header->queueCapacity, header->seqIn,
             header->seqOut);
    LOG_INFO("head: {}, tail: {}, buff-limit: {}", header->frontOffset, header->tailOffset, header->buffLimit);
    return lastPos;
}

uint8_t *ShmEnqueueHeadRaw(RmaShmHeader *header, int64_t dims[RMA_DIM_MAX], uint64_t sequence)
{
    int64_t dataSize = dims[0] * dims[1] * sizeof(float) * 1L;
    uint8_t *lastPos = nullptr;
    RmaShmData dataHead;
    uint64_t *out = (uint64_t *)(&header->buffLimit) + 1;

    LOG_INFO("before enqueue, capacity: {}, seq-in: {}, seq-out: {}, out:[ {} {} ]", header->queueCapacity,
             header->seqIn, header->seqOut, *out, *(out + 1));
    LOG_INFO("head: {}, tail: {}, buff-limit: {}", header->frontOffset, header->tailOffset, header->buffLimit);

    int64_t queueNum = header->seqIn - header->seqOut;
    if (queueNum >= header->queueCapacity) {
        LOG_ERROR("rma queue is full, num: {}", queueNum);
        return nullptr;
    }

    dataHead.totalLen = dataSize + RMA_SHM_DATA_HEAD;
    dataHead.dataType = 0;
    dataHead.dimNum = RMA_DIM_MAX;
    dataHead.dataLen = dataSize;
    dataHead.sequence = sequence;
    dataHead.dims[0] = dims[0];
    dataHead.dims[1] = dims[1];
    dataHead.readyLen = 0;

    if (header->tailOffset + dataSize > header->totalMemSize) {
        // 如果队列尾部的空余放不下新插入的数据，则从队列头部插入，当前约束队列不够大，不会被写满
        lastPos = reinterpret_cast<uint8_t *>(header) + RMA_SHM_HEAD_LEN;
        if (memcpy_s(lastPos, RMA_SHM_DATA_HEAD, &dataHead, RMA_SHM_DATA_HEAD) != EOK) {
            LOG_ERROR("memcpy failed");
            return nullptr;
        }

        header->buffLimit = header->tailOffset; // 标识该位置后面无可读取的数据，需要返回到队列首部
        header->tailOffset = RMA_SHM_HEAD_LEN + dataHead.totalLen; // 从队列首部开始偏移
    } else {
        // 正常顺序入队列
        lastPos = reinterpret_cast<uint8_t *>(header) + header->tailOffset;
        if (memcpy_s(lastPos, RMA_SHM_DATA_HEAD, &dataHead, RMA_SHM_DATA_HEAD) != EOK) {
            LOG_ERROR("memcpy failed");
            return nullptr;
        }
        header->tailOffset += dataHead.totalLen; // 尾部往后偏移，指导下一个元素的插入位置
    }

    header->seqIn = sequence; // 更新队列头的Seq

    LOG_INFO("after enqueue, capacity: {}, seq-in: {}, seq-out: {}", header->queueCapacity, header->seqIn,
             header->seqOut);
    LOG_INFO("head: {}, tail: {}, buff-limit: {}", header->frontOffset, header->tailOffset, header->buffLimit);
    return lastPos;
}

uint8_t *ShmEnqueueGetFront(RmaShmHeader *header, int64_t dims[RMA_DIM_MAX])
{
    int64_t dataSize = dims[0] * dims[1] * sizeof(float) * 1L;
    int64_t totalSize = dataSize + RMA_SHM_DATA_HEAD;
    uint8_t *lastPos = nullptr;

    LOG_INFO("before enqueue, capacity: {}, seq-in: {}, seq-out: {}", header->queueCapacity,
             header->seqIn, header->seqOut);
    LOG_INFO("head: {}, tail: {}, buff-limit: {}", header->frontOffset, header->tailOffset,
             header->buffLimit);

    int64_t queueNum = header->seqIn - header->seqOut;
    if (queueNum >= header->queueCapacity) {
        LOG_ERROR("rma queue is full, num: {}", queueNum);
        return nullptr;
    }

    if (header->tailOffset + totalSize > header->totalMemSize) {
        // 如果队列尾部的空余放不下新插入的数据，则从队列头部插入，当前约束队列不够大，不会被写满
        lastPos = reinterpret_cast<uint8_t *>(header) + RMA_SHM_HEAD_LEN;
    } else {
        // 正常顺序入队列
        lastPos = reinterpret_cast<uint8_t *>(header) + header->tailOffset;
    }

    return lastPos;
}

uint8_t *ShmEnqueueGetLast(RmaShmHeader *header, int64_t dims[RMA_DIM_MAX])
{
    int64_t dataSize = dims[0] * dims[1] * sizeof(float) * 1L;
    int64_t totalSize = dataSize + RMA_SHM_DATA_HEAD;
    uint8_t *lastPos = nullptr;

    int64_t queueNum = header->seqIn - header->seqOut;
    if (queueNum == 0) {
        return nullptr;
    }

    lastPos = reinterpret_cast<uint8_t *>(header) + (header->tailOffset - totalSize);

    return lastPos;
}

uint8_t *ShmEnqueue(RmaShmHeader *header, void *memData, int64_t memSize, uint64_t sequence)
{
    int64_t dataSize = memSize;
    uint8_t *lastPos = nullptr;

    LOG_INFO("before enqueue, capacity: {}, seq-in: {}, seq-out: {}", header->queueCapacity, header->seqIn,
             header->seqOut);
    LOG_INFO("head: {}, tail: {}, buff-limit: {}", header->frontOffset, header->tailOffset, header->buffLimit);

    int64_t queueNum = header->seqIn - header->seqOut;
    if (queueNum >= header->queueCapacity) {
        LOG_ERROR("rma queue is full, num: {}", queueNum);
        return nullptr;
    }

    if (header->tailOffset + dataSize > header->totalMemSize) {
        // 如果队列尾部的空余放不下新插入的数据，则从队列头部插入，当前约束队列不够大，不会被写满
        lastPos = reinterpret_cast<uint8_t *>(header) + RMA_SHM_HEAD_LEN;
        if (memcpy_s(lastPos, memSize, memData, memSize) != EOK) {
            LOG_ERROR("memcpy failed");
            return nullptr;
        }

        header->buffLimit = header->tailOffset; // 标识该位置后面无可读取的数据，需要返回到队列首部
        header->tailOffset = RMA_SHM_HEAD_LEN + dataSize; // 从队列首部开始偏移
    } else {
        // 正常顺序入队列
        lastPos = reinterpret_cast<uint8_t *>(header) + header->tailOffset;
        if (memcpy_s(lastPos, memSize, memData, memSize) != EOK) {
            LOG_ERROR("memcpy failed");
            return nullptr;
        }
        header->tailOffset += dataSize; // 尾部往后偏移，指导下一个元素的插入位置
    }

    header->seqIn = sequence; // 更新队列头的Seq

    LOG_INFO("after enqueue, capacity: {}, seq-in: {}, seq-out: {}", header->queueCapacity, header->seqIn,
             header->seqOut);
    LOG_INFO("head: {}, tail: {}, buff-limit: {}", header->frontOffset, header->tailOffset, header->buffLimit);
    return lastPos;
}

int64_t GetShmElemNum(RmaShmHeader *header)
{
    int64_t queueNum = header->seqIn - header->seqOut;

    return queueNum;
}


uint8_t *ShmOutqueue(RmaShmHeader *header)
{
    RmaShmHeader rmaHeader;
    uint64_t dataLen = 0;

    if (memcpy_s(&rmaHeader, sizeof(RmaShmHeader), header, sizeof(RmaShmHeader)) != EOK) {
        LOG_ERROR("memcpy failed");
        return nullptr;
    }

    if (GetShmElemNum(&rmaHeader) <= 0) {
        return nullptr;
    }

    LOG_INFO("before outqueue, capacity: {}, seq-in: {}, seq-out: {}", rmaHeader.queueCapacity, rmaHeader.seqIn,
             rmaHeader.seqOut);
    LOG_INFO("offset: {}, tail: {}, buff-limit: {}", rmaHeader.frontOffset, rmaHeader.tailOffset, rmaHeader.buffLimit);

    uint8_t *lastPos = nullptr;
    if (rmaHeader.frontOffset == rmaHeader.buffLimit) { // 队列尾部的数据都已读取完，需要返回到首部
        header->frontOffset = RMA_SHM_HEAD_LEN;
        header->buffLimit = 0;
        lastPos = reinterpret_cast<uint8_t *>(header) + RMA_SHM_HEAD_LEN;
    } else {
        lastPos = reinterpret_cast<uint8_t *>(header) + rmaHeader.frontOffset;
    }

    if (memcpy_s(&dataLen, sizeof(uint64_t), lastPos, sizeof(uint64_t)) != EOK) {
        LOG_ERROR("memcpy failed");
        return nullptr;
    }
    header->frontOffset += dataLen;

    LOG_INFO("after outqueue, seq-in: {}, seq-out: {}, data-len: {}", header->seqIn, header->seqOut, dataLen);
    LOG_INFO("offset: {}, tail: {}, buff-limit: {}", header->frontOffset, header->tailOffset, header->buffLimit);

    return lastPos;
}

void SetShmQueueSeqOut(RmaShmHeader *header, uint64_t sequence)
{
    header->seqOut = sequence;
    LOG_DEBUG("queue seq-out: {}", header->seqOut);
}

void SetShmQueueSeqIn(RmaShmHeader *header, uint64_t sequence)
{
    header->seqIn = sequence;
    LOG_DEBUG("queue seq-out: {}", header->seqIn);
}