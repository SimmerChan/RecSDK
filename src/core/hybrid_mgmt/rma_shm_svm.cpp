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
#include <cstdio>
#include "securec.h"
#include <sys/stat.h>
#include <sys/shm.h>
#include <driver/ascend_hal_define.h>
#include <unordered_map>
#include "rma_shm_svm.h"
#include "utils/common.h"

using namespace MxRec;
using namespace std;

extern "C" {
drvError_t halHostRegister(void *srcPtr, UINT64 size, UINT32 flag, UINT32 devid, void **dstPtr);
drvError_t halHostUnregister(void *srcPtr, UINT32 devid);
drvError_t rtDeviceGetBareTgid(uint32_t *pid);
}

const uint64_t RMA_SHM_TOTAL_MEM_SIZE = 1 * 1024 * 1024 * 1024 * 1L; // 共享内存总容量 单位B
constexpr int RMA_SHM_QUEUE_CAPACITY = 50;                           // 队列的最大深度
constexpr int32_t MAX_RANK_SIZE = 4095;

uint32_t g_pid = 0;
bool g_aclInit[MAX_RANK_SIZE] = {false};
std::unordered_map<std::string, void *> g_shmSvmMap;
std::unordered_map<std::string, void *> g_shmAddr;
std::unordered_map<std::string, int> g_shmId;

RmaDevModel g_rmaDevModel = RmaDevModel::PCIE_TH_DEV;

void InitShmHeader(RmaShmHeader *header, int64_t memSize, int32_t capacity)
{
    header->totalMemSize = memSize - RMA_SHM_HEAD_LEN;
    header->queueCapacity = capacity;
    header->seqIn = 0;
    header->seqOut = 0;
    header->frontOffset = RMA_SHM_HEAD_LEN;
    header->tailOffset = RMA_SHM_HEAD_LEN;
    header->buffLimit = 0;
}

void ResetShmHeader(RmaShmHeader *header)
{
    header->seqIn = 0;
    header->seqOut = 0;
    header->frontOffset = RMA_SHM_HEAD_LEN;
    header->tailOffset = RMA_SHM_HEAD_LEN;
    header->buffLimit = 0;
}

void RmaFreeShm(std::string shmName, void *memory)
{
    if (g_rmaDevModel == RmaDevModel::SVM_MAP_DEV) {
        if (aclrtFreeHost(memory) != ACL_ERROR_NONE) {
            LOG_WARN("Free host mem failed.");
        }
    } else {
        int shmId = g_shmId[shmName];
        LOG_INFO("Free shm with shmid: {} success.", shmId);
        (void)shmdt(memory);

        shmctl(shmId, IPC_RMID, nullptr);
        g_shmId.erase(shmName);
    }
}

bool IsPrefix(const std::string& str, const std::string& prefix)
{
    if (prefix.length() > str.length()) {
        return false;
    }
    return str.compare(0, prefix.length(), prefix) == 0;
}

uint32_t GetRegisterFlag(RmaDevModel mode)
{
    switch (mode) {
        case RmaDevModel::MEM_MAP_DEV:
            return HOST_MEM_MAP_DEV;
        case RmaDevModel::SVM_MAP_DEV:
            return HOST_SVM_MAP_DEV;
        default:
            return HOST_MEM_MAP_DEV_PCIE_TH;
    }
}

// aicore申请shm内存
void *RmaCreateShm(std::string shmName, uint64_t memSize, int deviceId, int capacity)
{
    string chipName = GetChipName(deviceId);
    if (IsPrefix(chipName, "910B")) {
        g_rmaDevModel = RmaDevModel::PCIE_TH_DEV;
    } else if (IsPrefix(chipName, "910_93")) {
        g_rmaDevModel = RmaDevModel::SVM_MAP_DEV;
    } else {
        auto error = Error(ModuleName::M_RMA_SHM_SVM, ErrorType::UNKNOWN,
                           StringFormat("Unsupported chip type: %s.", chipName.c_str()));
        LOG_ERROR(error.ToString());
        throw runtime_error(error.ToString());
    }
    void *memory = nullptr;
    if (g_rmaDevModel == RmaDevModel::SVM_MAP_DEV) {
        if (aclrtMallocHost((void **)&memory, memSize) != ACL_ERROR_NONE) {
            auto error = Error(ModuleName::M_RMA_SHM_SVM, ErrorType::UNKNOWN,
                               StringFormat("Malloc host memory failed."));
            LOG_ERROR(error.ToString());
            throw runtime_error(error.ToString());
        }
        (void)aclrtMemset(memory, memSize, 0, memSize);
        LOG_INFO("Create memory {}, size: {} bytes successfully.", shmName.c_str(), memSize);
    } else {
        struct shmid_ds buf;
        key_t key = static_cast<key_t>(std::hash<std::string> {}(shmName));
        int shmId = -1;
        if (GlobalEnv::hugeTlbEnable) {
            shmId = shmget(key, memSize, IPC_CREAT | 0666 | SHM_HUGETLB);
        } else {
            shmId = shmget(key, memSize, IPC_CREAT | 0600);
        }
        if (shmId == -1) {
            auto error = Error(ModuleName::M_RMA_SHM_SVM, ErrorType::UNKNOWN,
                               StringFormat("Shmget failed."));
            LOG_ERROR(error.ToString());
            throw runtime_error(error.ToString());
        }

        memory = shmat(shmId, nullptr, 0);
        if (memory == (void *)-1) {
            shmctl(shmId, IPC_RMID, nullptr);
            auto error = Error(ModuleName::M_RMA_SHM_SVM, ErrorType::UNKNOWN,
                               StringFormat("Shmat failed."));
            LOG_ERROR(error.ToString());
            throw runtime_error(error.ToString());
        }

        shmctl(shmId, IPC_STAT, &buf);
        (void)memset_s(memory, memSize, 0, memSize);
        g_shmId.insert(std::make_pair(shmName, shmId));
        LOG_INFO("Create shm {}, shmid: {}, size: {} bytes successfully.", shmName.c_str(), shmId, memSize);
    }

    uint32_t flag = GetRegisterFlag(g_rmaDevModel);
    void *svmMem = nullptr;
    if (halHostRegister(memory, memSize, flag, deviceId, &svmMem) != DRV_ERROR_NONE) {
        RmaFreeShm(shmName, memory);
        auto error = Error(ModuleName::M_RMA_SHM_SVM, ErrorType::UNKNOWN,
                           StringFormat("Device %d halHostRegister failed.", deviceId));
        LOG_ERROR(error.ToString());
        throw runtime_error(error.ToString());
    }

    g_shmAddr.insert(std::make_pair(shmName, memory));

    // 初始化队列头
    InitShmHeader(reinterpret_cast<RmaShmHeader *>(memory), memSize, capacity);

    return svmMem;
}

// 仅用于pybind侧调用，创建共享内存
int64_t GetShmAddr(std::string name, int deviceId, int capacity)
{
    auto memSize = RMA_SHM_TOTAL_MEM_SIZE;
    if (capacity > RMA_SHM_QUEUE_CAPACITY) {
        capacity = RMA_SHM_QUEUE_CAPACITY;
    }

    if (!g_aclInit[deviceId]) {
        aclInit(nullptr);
        aclrtSetDevice(deviceId);
        g_aclInit[deviceId] = true;
    }

    if (g_pid == 0) {
        (void)rtDeviceGetBareTgid(&g_pid);
    }
    std::string shmName = name + "_" + std::to_string(g_pid);
    LOG_INFO("Device {} start alloc shm for {}.", deviceId, shmName.c_str());

    if (g_shmSvmMap.find(shmName) == g_shmSvmMap.end()) {
        g_shmSvmMap.insert(std::make_pair(shmName, RmaCreateShm(shmName, memSize, deviceId, capacity)));
    }

    return reinterpret_cast<int64_t>(g_shmSvmMap[shmName]);
}

// 仅用于hd_transfer的send/recv调用，获取host侧地址；
void *GetHostAddr(std::string name)
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
        LOG_INFO("Device {} free memory: {} success.", deviceId, pair.first.c_str());
    }
    g_shmAddr.clear();
    g_shmId.clear();
    g_shmSvmMap.clear();

    if (g_aclInit[deviceId]) {
        aclrtResetDevice(deviceId);
        aclFinalize();
        g_aclInit[deviceId] = false;
        LOG_INFO("Device {} free acl device.", deviceId);
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
        ResetShmHeader(reinterpret_cast<RmaShmHeader *>(pair.second));
        LOG_INFO("Reset queue: {}", pair.first);
    }
}

uint8_t *ShmEnqueueHeadRaw(RmaShmHeader *header, int64_t dims[RMA_DIM_MAX], uint64_t sequence)
{
    int64_t dataSize = dims[0] * dims[1] * sizeof(float) * 1L;
    uint8_t *lastPos = nullptr;
    RmaShmData dataHead;

    LOG_INFO("Before enqueue, capacity: {}, seq-in: {}, seq-out: {}, head: {}, tail: {}, buff-limit: {}.",
             header->queueCapacity, header->seqIn, header->seqOut,
             header->frontOffset, header->tailOffset, header->buffLimit);

    while (header->seqIn - header->seqOut >= header->queueCapacity) {
        this_thread::sleep_for(1ms);
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
            auto error = Error(ModuleName::M_RMA_SHM_SVM, ErrorType::UNKNOWN,
                               StringFormat("Data head memcpy failed."));
            LOG_ERROR(error.ToString());
            throw runtime_error(error.ToString());
        }
        header->buffLimit = header->tailOffset; // 标识该位置后面无可读取的数据，需要返回到队列首部
        header->tailOffset = RMA_SHM_HEAD_LEN + dataHead.totalLen; // 从队列首部开始偏移
    } else {
        // 正常顺序入队列
        lastPos = reinterpret_cast<uint8_t *>(header) + header->tailOffset;
        if (memcpy_s(lastPos, RMA_SHM_DATA_HEAD, &dataHead, RMA_SHM_DATA_HEAD) != EOK) {
            auto error = Error(ModuleName::M_RMA_SHM_SVM, ErrorType::UNKNOWN,
                               StringFormat("Data head memcpy failed."));
            LOG_ERROR(error.ToString());
            throw runtime_error(error.ToString());
        }
        header->tailOffset += dataHead.totalLen; // 尾部往后偏移，指导下一个元素的插入位置
    }

    header->seqIn = sequence; // 更新队列头的Seq

    LOG_INFO("After enqueue, capacity: {}, seq-in: {}, seq-out: {}, head: {}, tail: {}, buff-limit: {}.",
             header->queueCapacity, header->seqIn, header->seqOut,
             header->frontOffset, header->tailOffset, header->buffLimit);
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
        auto error = Error(ModuleName::M_RMA_SHM_SVM, ErrorType::UNKNOWN,
                           StringFormat("Queue head memcpy failed."));
        LOG_ERROR(error.ToString());
        throw runtime_error(error.ToString());
    }

    if (GetShmElemNum(&rmaHeader) <= 0) {
        return nullptr;
    }

    LOG_INFO("Before outqueue, capacity: {}, seq-in: {}, seq-out: {}, offset: {}, tail: {}, buff-limit: {}.",
             rmaHeader.queueCapacity, rmaHeader.seqIn, rmaHeader.seqOut,
             rmaHeader.frontOffset, rmaHeader.tailOffset, rmaHeader.buffLimit);

    uint8_t *lastPos = nullptr;
    if (rmaHeader.frontOffset == rmaHeader.buffLimit) { // 队列尾部的数据都已读取完，需要返回到首部
        header->frontOffset = RMA_SHM_HEAD_LEN;
        header->buffLimit = 0;
        lastPos = reinterpret_cast<uint8_t *>(header) + RMA_SHM_HEAD_LEN;
    } else {
        lastPos = reinterpret_cast<uint8_t *>(header) + rmaHeader.frontOffset;
    }

    dataLen = ((RmaShmData *)lastPos)->totalLen;
    header->frontOffset += dataLen;

    LOG_INFO("After outqueue, seq-in: {}, seq-out: {}, data-len: {}, offset: {}, tail: {}, buff-limit: {}.",
             header->seqIn, header->seqOut, dataLen, header->frontOffset, header->tailOffset, header->buffLimit);

    return lastPos;
}

void SetShmQueueSeqOut(RmaShmHeader *header, uint64_t sequence)
{
    header->seqOut = sequence;
}
