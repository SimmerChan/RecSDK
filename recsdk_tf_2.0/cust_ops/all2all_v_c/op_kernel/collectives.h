/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#ifndef LCCL_COLLECTIVES_H
#define LCCL_COLLECTIVES_H

#include "sync_collectives.h"
using namespace AscendC;

#define KERNELS_ARGS_FUN() \
GM_ADDR input, GM_ADDR send_count_matrix, GM_ADDR shape_vec, GM_ADDR peer_mem,    \
GM_ADDR output, int64_t rank, int64_t rankSize, int64_t magic

#define KERNELS_ARGS_CALL() input, send_count_matrix, shape_vec, peer_mem, output, rank, rankSize, magic

#define KERNELS_ARGS_CF_FUN() \
GM_ADDR input, GM_ADDR send_count_matrix, GM_ADDR shape_vec, GM_ADDR peer_mem,    \
GM_ADDR output, int64_t rank, int64_t rankSize, int64_t localRankSize, int64_t magic, int dim, int outShape

#define KERNELS_ARGS_CF_CALL() input, send_count_matrix, shape_vec, peer_mem, output, rank, rankSize,    \
localRankSize, magic, dim, outShape

class Collectives {
    constexpr static int32_t UB_ALIGN_SIZE = 32;
    constexpr static int32_t UB_SINGLE_TOTAL_SIZE_MAX = 192 * 1024;
    constexpr static int32_t UB_HEAD_OFFSET = 96;
    constexpr static int32_t UB_MID_OFFSET = UB_HEAD_OFFSET + UB_SINGLE_PING_PONG_ADD_SIZE_MAX + UB_ALIGN_SIZE;
public:
    FORCE_INLINE_AICORE Collectives(int rank, int rankSize, uint32_t extraFlag)
        : rank(rank), rankSize(rankSize), extraFlag(extraFlag) {}

    FORCE_INLINE_AICORE void Init(KERNELS_ARGS_FUN())
    {
        this->root = 0;
        this->len = 0;
        this->magic = magic;
        this->rank = rank;
        this->rankSize = rankSize;

        blockIdx = GetBlockIdx();
        blockNum = GetBlockNum();

        GlobalTensor<int64_t> peerMemsAddrGm;
        int64_t peer_mem_addr = reinterpret_cast<int64_t>(peer_mem);
        peerMemsAddrGm.SetGlobalBuffer((__gm__ int64_t*)peer_mem_addr, rankSize * sizeof (int64_t));
        for (int i = 0; i < rankSize; ++i) {
            shareAddrs[i] = (GM_ADDR)(peerMemsAddrGm.GetValue(i))+
                            (this->magic % PING_PONG_SIZE) * (IPC_BUFF_MAX_SIZE + IPC_DATA_OFFSET);
        }

        pipe.InitBuffer(tBuf, UB_SINGLE_TOTAL_SIZE_MAX);
        sync.Init(rank, rankSize, shareAddrs, tBuf);
    }

public:
    template <typename T>
    FORCE_INLINE_AICORE void SetAtomicOpType(int op)
    {
        switch (op) {
            case ADD:
                AscendC::SetAtomicAdd<T>();
                break;
            case MUL:
                // 累乘时忽略不设置atomic寄存器
                break;
            case MAX:
                AscendC::SetAtomicMax<T>();
                break;
            case MIN:
                AscendC::SetAtomicMin<T>();
                break;
            default:
                AscendC::SetAtomicNone();
        }
    }

    template <typename T>
    FORCE_INLINE_AICORE void SetAtomic(int op)
    {
        PipeBarrier<PIPE_ALL>();
        if (op != -1) {
            SetAtomicOpType<T>(op);
        }
        PipeBarrier<PIPE_ALL>();
    }

    template <typename T>
    FORCE_INLINE_AICORE void CpGM2GMPingPong(int64_t dataSizeRemain, const GlobalTensor<T>& inputGT,
                                             const GlobalTensor<T>& outputGT, int op)
    {
        if (dataSizeRemain <= 0) {
            return;
        }
        __gm__ T *input = const_cast<__gm__ T *>(inputGT.GetPhyAddr());
        __gm__ T *output = const_cast<__gm__ T *>(outputGT.GetPhyAddr());
        // each UB's size is 95 KB
        const int64_t firstUBStart = UB_HEAD_OFFSET;
        const int64_t secondUBStart = UB_MID_OFFSET;
        __ubuf__ T* inputUB[2] = {(__ubuf__ T*)get_imm(firstUBStart), (__ubuf__ T*)get_imm(secondUBStart)};
        int inputOffsetNum = 0;
        int outputOffsetNum = 0;

        SetAtomic<T>(op);

        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);    // MTE2等MTE3
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);    // MTE2等MTE3
        for (int64_t i = 0; dataSizeRemain > 0; i++) {
            uint32_t size = dataSizeRemain > UB_SINGLE_PING_PONG_ADD_SIZE_MAX ?
                            UB_SINGLE_PING_PONG_ADD_SIZE_MAX : dataSizeRemain;
            event_t eventId = (i & 1) ? EVENT_ID0 : EVENT_ID1;
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventId);
            CpGM2UB((i & 1) ? inputUB[0] : inputUB[1], input + inputOffsetNum, size);
            AscendC::SetFlag<HardEvent::MTE2_MTE3>(eventId);
            AscendC::WaitFlag<HardEvent::MTE2_MTE3>(eventId);
            CpUB2GM(output + outputOffsetNum, (i & 1) ? inputUB[0] : inputUB[1], size);
            AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventId);

            dataSizeRemain -= size;
            inputOffsetNum += (size / sizeof(T));
            outputOffsetNum += (size / sizeof(T));
        }
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);    // MTE2等MTE3
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);    // MTE2等MTE3

        AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID3);    // Scalar等MTE3
        AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
        if (op != COPYONLY) {
            AscendC::SetAtomicNone();
        }
        PipeBarrier<PIPE_ALL>();
    }

protected:
    int rank;
    int rankSize;
    int localRankSize;
    int xRankSize;
    int yRankSize;
    int xRankIdx;
    int yRankIdx;
    uint32_t extraFlag;
    int root;
    int64_t len;
    int64_t magic;
    int64_t blockIdx;  // 当前aicore序号
    int64_t blockNum;  // 当前rank的总aicore数
    GM_ADDR shareAddrs[LCAL_MAX_RANK_SIZE];  // 共享内存地址列表
    TPipe pipe;  // pipe工具类
    TBuf<QuePosition::VECCALC> tBuf;
    SyncCollectives sync;
    int dim;

private:
    template <typename T>
    FORCE_INLINE_AICORE void CpUB2GM(__gm__ T *gmAddr, __ubuf__ T *ubAddr, uint32_t size)
    {
#if defined(__DAV_C310) || defined(__DAV_M310) || defined(__DAV_L310) || defined(__DAV_L311)
        copy_ubuf_to_gm_align_v2((__gm__ uint32_t *)gmAddr, (__ubuf__ uint32_t *)ubAddr, 0, 1, size, 0, 0, 0);
#else
        LocalTensor<uint8_t> ubTensor;
        GlobalTensor<uint8_t> gmTensor;
        DataCopyExtParams dataCopyParams(1, size, 0, 0, 0);
        ubTensor.address_.logicPos = static_cast<uint8_t>(TPosition::VECIN);
        ubTensor.address_.bufferAddr = reinterpret_cast<uint64_t>(ubAddr);
        gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(gmAddr));
        DataCopyPad(gmTensor, ubTensor, dataCopyParams);
#endif
    }

    template <typename T>
    FORCE_INLINE_AICORE void CpGM2UB(__ubuf__ T *ubAddr, __gm__ T *gmAddr, uint32_t size)
    {
#if defined(__DAV_C310) || defined(__DAV_M310) || defined(__DAV_L310) || defined(__DAV_L311)
        copy_gm_to_ubuf_align_v2((__ubuf__ uint32_t *)ubAddr, (__gm__ uint32_t *)gmAddr, 0, 1, size, 0, 0, 0, 0, 0, 0);
#else
        LocalTensor<uint8_t> ubTensor;
        GlobalTensor<uint8_t> gmTensor;
        DataCopyExtParams dataCopyParams(1, size, 0, 0, 0);
        ubTensor.address_.logicPos = static_cast<uint8_t>(TPosition::VECIN);
        ubTensor.address_.bufferAddr = reinterpret_cast<uint64_t>(ubAddr);
        gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(gmAddr));
        DataCopyPadExtParams<uint8_t> padParams;
        DataCopyPad(ubTensor, gmTensor, dataCopyParams, padParams);
#endif
    }
};

// CeilDiv
template <typename T1, typename T2>
FORCE_INLINE_AICORE T1 CeilDiv(T1 a, T2 b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

#endif  // LCCL_COLLECTIVES_H