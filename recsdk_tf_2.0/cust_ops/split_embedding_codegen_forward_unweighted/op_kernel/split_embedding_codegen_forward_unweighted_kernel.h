/* Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.

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

#ifndef SPLIT_EMBEDDING_CODEGEN_FORWARD_UNWEIGHTED_KERNEL_FUN_H
#define SPLIT_EMBEDDING_CODEGEN_FORWARD_UNWEIGHTED_KERNEL_FUN_H

#include <cstdint>

#include "kernel_operator.h"

using namespace AscendC;

namespace SplitEmbeddingCodegenForwardUnweighted {

constexpr int USE_QUEUE_NUM = 2;
constexpr int DATA_ALIGN_BYTES = 32;
constexpr int DATA_TYPE_INT64 = 1;
constexpr int FLOAT_ALIGNMENT = 8;
constexpr int DATA_TYPE_FLOAT32 = 0;
constexpr int SUM_POOL = 0;
constexpr int MEAN_POOL = 1;
constexpr int MAX_INDICS_ONE_BLOCK = 100;

struct Args {
    GM_ADDR devWeights;
    GM_ADDR weightsPlacements;
    GM_ADDR weightsOffsets;
    GM_ADDR dOffsets;
    GM_ADDR indices;
    GM_ADDR offsets;
    GM_ADDR out;
    GM_ADDR tiling;
    GM_ADDR workspace;
};

struct ComputeArgs {
    int64_t offsetIndex;
    int64_t embedDim;
    int64_t indWeightOffset;
    int64_t outOffset;
};

__aicore__ inline int64_t GetOffset(GM_ADDR offsetAddr, int64_t index, int64_t datType)
{
    __gm__ int64_t* offsetPtr = (__gm__ int64_t*)offsetAddr;
    return *(offsetPtr + index);
}

class SplitEmbeddingCodegenForwardUnweightedKernel {
public:
    __aicore__ inline SplitEmbeddingCodegenForwardUnweightedKernel(Args args)
    {
        GET_TILING_DATA(tilingData, args.tiling);
        // ADDR
        devWeights = args.devWeights;
        weightsPlacements = args.weightsPlacements;
        weightsOffsets = args.weightsOffsets;
        dOffsets = args.dOffsets;
        indices = args.indices;
        offsets = args.offsets;
        out = args.out;
        workspace = args.workspace;

        // Shape
        devWeightsDim0 = tilingData.devWeightsDim0;
        weightsOffsetsDim0 = tilingData.weightsOffsetsDim0;
        dOffsetsDim0 = tilingData.dOffsetsDim0;
        indicesDim0 = tilingData.indicesDim0;
        offsetsDim0 = tilingData.offsetsDim0;
        outDim0 = tilingData.outDim0;
        outDim1 = tilingData.outDim1;
        maxD = tilingData.maxD;

        // DataType
        bytesOfDataType = sizeof(float);
        offsetDataType = DATA_TYPE_INT64;

        // Tiling
        splitBaseLen = tilingData.splitBaseLen;
        tailSplitIndex = tilingData.tailSplitIndex;

        // Ub
        ubCanUsed = tilingData.ubCanUsed;
        blockLen = ubCanUsed / USE_QUEUE_NUM / bytesOfDataType;
        blockLen = blockLen / FLOAT_ALIGNMENT * FLOAT_ALIGNMENT;

        // func
        poolMode = tilingData.poolMode;
        // ThisCoreLen
        if (GetBlockIdx() >= tailSplitIndex) {
            lenOfThisCore = splitBaseLen;
            offsetOfThisCore = tailSplitIndex * (splitBaseLen + 1) + (GetBlockIdx() - tailSplitIndex) * splitBaseLen;
        } else {
            lenOfThisCore = splitBaseLen + 1;
            offsetOfThisCore = GetBlockIdx() * (splitBaseLen + 1);
        }

        devWeightsGT.SetGlobalBuffer((__gm__ float*)devWeights, devWeightsDim0);
        outGT.SetGlobalBuffer((__gm__ float*)out, outDim0 * outDim1);

        // Init pipe
        pipe.InitBuffer(queIn, 1, blockLen * sizeof(float));
        pipe.InitBuffer(queOut, 1, blockLen * sizeof(float));
    }

    template <typename T>
    __aicore__ inline void CpGm2Local(const LocalTensor<T>& lt, const GlobalTensor<T>& gt, int64_t len)
    {
        uint32_t alignLen = len * sizeof(T) / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES;
        uint32_t unAlignLen = len * sizeof(T) - alignLen;

        DataCopy(lt, gt, alignLen / sizeof(T));
        if (unAlignLen != 0) {
            const DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
            const DataCopyPadExtParams<float> dataCopyPadExtParams{false, 0, 0, 0};
            DataCopyPad(lt[alignLen / sizeof(T)], gt[alignLen / sizeof(T)], dataCopyExtParams, dataCopyPadExtParams);
        }
    }

    template <typename T>
    __aicore__ inline void CpLocal2Gm(const GlobalTensor<T>& gt, const LocalTensor<T>& lt, int64_t len)
    {
        uint32_t alignLen = len * sizeof(T) / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES;
        uint32_t unAlignLen = len * sizeof(T) - alignLen;
        DataCopy(gt, lt, alignLen / sizeof(T));
        if (unAlignLen != 0) {
            const DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
            DataCopyPad(gt[alignLen / sizeof(T)], lt[alignLen / sizeof(T)], dataCopyExtParams);
        }
    }

    __aicore__ inline void Compute()
    {
        __gm__ int32_t* dOffsetsPtr = (__gm__ int32_t*)dOffsets;
        __gm__ int64_t* weightsOffsetsPtr = (__gm__ int64_t*)weightsOffsets;
        __gm__ int64_t* offsetsPtr = (__gm__ int64_t*)offsets;
        __gm__ float* x = (__gm__ float*)out;
        int64_t thisOffsetIndex = 0;
        for (int64_t i = offsetsDim0 - 1; i >= 0; i--) {
            if (offsetOfThisCore >= *(offsetsPtr + i)) {
                thisOffsetIndex = i;
                break;
            }
        }

        int64_t total = lenOfThisCore;
        int64_t remain = total;
        int64_t indicesNumOneBlock = blockLen / maxD;

        ComputeArgs argsArry[300];
        while (remain > 0) {
            int64_t thisLen = indicesNumOneBlock;
            if (remain < thisLen) {
                thisLen = remain;
            }
            int64_t thisOffset = offsetOfThisCore + total - remain;

            // Compute args
            for (int64_t i = 0; i < thisLen; i++) {
                int64_t indicesInd = i + thisOffset;
                if (indicesInd >= *(offsetsPtr + thisOffsetIndex + 1)) {
                    thisOffsetIndex = thisOffsetIndex + 1;
                }

                // Which Table Used, and the table embedDim
                int64_t offsetIndex = thisOffsetIndex;
                int64_t tableIndex = thisOffsetIndex / outDim0;

                // Which weight offset
                int64_t embedDim = *(dOffsetsPtr + tableIndex + 1) - *(dOffsetsPtr + tableIndex);
                int64_t thisWeightOffset = *(weightsOffsetsPtr + tableIndex);

                int64_t thisIndForThisTable = GetOffset(indices, indicesInd, offsetDataType);
                int64_t indWeightOffset = thisIndForThisTable * embedDim + thisWeightOffset;

                // batcheSize
                int64_t outBatchInd = thisOffsetIndex % outDim0;
                int64_t outEmbedOffset = *(dOffsetsPtr + tableIndex);
                int64_t outOffset = outBatchInd * outDim1 + outEmbedOffset;

                ComputeArgs& theArgs = argsArry[i];
                theArgs.offsetIndex = offsetIndex;
                theArgs.embedDim = embedDim;
                theArgs.indWeightOffset = indWeightOffset;
                theArgs.outOffset = outOffset;
            }

            // Copy In
            LocalTensor<float> inputLt = queIn.AllocTensor<float>();
            for (int64_t i = 0; i < thisLen; i++) {
                ComputeArgs theArgs = argsArry[i];
                CpGm2Local(inputLt[i * maxD], devWeightsGT[theArgs.indWeightOffset], theArgs.embedDim);
            }
            queIn.EnQue(inputLt);

            LocalTensor<float> newInputLt = queIn.DeQue<float>();
            LocalTensor<float> outLt = queOut.AllocTensor<float>();

            // Compute
            if (poolMode == MEAN_POOL) {
                for (int64_t i = 0; i < thisLen; i++) {
                    ComputeArgs theArgs = argsArry[i];
                    int64_t thisBagLen = *(offsetsPtr + theArgs.offsetIndex + 1) - *(offsetsPtr + theArgs.offsetIndex);
                    float meanLen = (float)1 / thisBagLen;
                    Muls<float>(outLt[i * maxD], newInputLt[i * maxD], meanLen, maxD);
                }
            } else {
                DataCopy(outLt, newInputLt, blockLen);
            }

            queOut.EnQue(outLt);
            queIn.FreeTensor(newInputLt);

            // Copy Out
            LocalTensor<float> newOutLt = queOut.DeQue<float>();
            SetAtomicAdd<float>();
            for (int64_t i = 0; i < thisLen; i++) {
                ComputeArgs theArgs = argsArry[i];
                CpLocal2Gm(outGT[theArgs.outOffset], newOutLt[i * maxD], theArgs.embedDim);
            }
            SetAtomicNone();
            queOut.FreeTensor(newOutLt);

            remain = remain - thisLen;
        }
    }

private:
    // // GM_ADDR
    GM_ADDR devWeights;
    GM_ADDR weightsPlacements;
    GM_ADDR weightsOffsets;
    GM_ADDR dOffsets;
    GM_ADDR indices;
    GM_ADDR offsets;
    GM_ADDR out;
    GM_ADDR workspace;

    // Shape
    int64_t devWeightsDim0;
    int64_t weightsOffsetsDim0;
    int64_t dOffsetsDim0;
    int64_t indicesDim0;
    int64_t offsetsDim0;
    int64_t outDim0;
    int64_t outDim1;
    int64_t maxD;

    // // DataType
    int64_t bytesOfDataType;
    int64_t offsetDataType;

    // Tiling
    int64_t splitBaseLen;
    int64_t tailSplitIndex;

    // Ub
    int64_t ubCanUsed;
    int64_t blockLen;

    // func change
    int64_t poolMode;

    // ThisCoreLen
    int64_t lenOfThisCore;
    int64_t offsetOfThisCore;

    // Tpipe
    TPipe pipe;
    TQue<TPosition::VECIN, 1> queIn;
    TQue<TPosition::VECOUT, 1> queOut;

    // ThisCoreAddr
    GlobalTensor<float> devWeightsGT;
    GlobalTensor<float> outGT;
};
}  // namespace SplitEmbeddingCodegenForwardUnweighted
#endif
