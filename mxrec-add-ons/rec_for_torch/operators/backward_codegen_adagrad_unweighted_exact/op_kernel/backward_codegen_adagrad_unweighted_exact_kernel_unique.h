/**
 * @file backward_codegen_adagrad_unweighted_exact_kernel_unique.h
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 */

#ifndef BACKWARD_CODEGEN_ADAGRAD_UNWEIGHTED_EXACT_KERNEL_UNIQUE_FUN_H
#define BACKWARD_CODEGEN_ADAGRAD_UNWEIGHTED_EXACT_KERNEL_UNIQUE_FUN_H

#include <cstdint>

#include "kernel_operator.h"
#include "backward_codegen_unweighted_exact_kernel_unique.h"

using namespace AscendC;
using namespace BackwardCodegenUnweightedExactUnique;
using namespace BackwardCodegenUnweightedExact;
namespace BackwardCodegenAdagradUnweightedExactUnique {

constexpr int NUM_OF_OUT = 2;

class BackwardCodegenAdagradUnweightedExactKernelUnique : public BackwardCodegenUnweightedExactKernelUnique {
public:
    __aicore__ inline BackwardCodegenAdagradUnweightedExactKernelUnique() {}
    __aicore__ inline void AdaScheduler()
    {
        int64_t lastIndices = 0;
        for (int64_t i = 1; i < uniqueHashDim0; i++) {
            if (uniqueHashSizeGT.GetValue(i) != lastIndices) {
                Scheduler(uniqueHashSizeGT.GetValue(i) - lastIndices, offsetOfThisCore, lenOfThisCore);
                if (lenOfThisCore > 0) {
                    UpdateEmbedAda(offsetOfThisCore + lastIndices, lenOfThisCore, i - 1);
                }
                lastIndices = uniqueHashSizeGT.GetValue(i);
            }
        }
    }

    __aicore__ inline void ComputeAda(LocalTensor<float>inputLt, LocalTensor<float>outLt, int64_t thisLen)
    {
        int64_t momentum1Offset = thisLen * maxD;
        Mul<float>(outLt, inputLt, inputLt, momentum1Offset);
        Add<float>(outLt, inputLt[momentum1Offset], outLt, momentum1Offset);
        Sqrt<float>(outLt, outLt, momentum1Offset);
        Adds<float>(outLt, outLt, eps, momentum1Offset);
        Duplicate<float>(outLt[momentum1Offset], learning_rate, momentum1Offset);
        Div<float>(outLt, outLt[momentum1Offset], outLt, momentum1Offset);
        Mul<float>(outLt, outLt, inputLt, momentum1Offset);
        Muls<float>(outLt, outLt, -1, momentum1Offset);
        Mul<float>(outLt[momentum1Offset], inputLt, inputLt, momentum1Offset);
    }

    __aicore__ inline void UpdateEmbedAda(int64_t offsetLen, int64_t totalLen, int64_t tableIndex)
    {
        __gm__ int32_t* dOffsetsPtr = (__gm__ int32_t*)dOffsets;
        __gm__ int64_t* weightsOffsetsPtr = (__gm__ int64_t*)weightsOffsets;
        int indicesNumOneBlock = blockLen / NUM_OF_OUT / maxD;
        if (indicesNumOneBlock >= MAX_ARGS_PIPE_LEN) {
            indicesNumOneBlock = MAX_ARGS_PIPE_LEN;
        }
        int64_t remain = totalLen;
        int64_t thisLen = remain;
        while (remain > 0) {
            if (remain > indicesNumOneBlock) {
                thisLen = indicesNumOneBlock;
            }
            remain -= thisLen;
            LocalTensor<float> inputLt = queIn.AllocTensor<float>();
            LocalTensor<float> outputLt = queOut.AllocTensor<float>();
            LocalTensor<int64_t> indicesLt = queIndices.AllocTensor<int64_t>();
            
            // copyIn
            CpGm2Local(indicesLt, uniqueIdGT[offsetLen], thisLen);
            CpGm2Local(inputLt, outGT[offsetLen * maxD], thisLen * maxD);
            int64_t outIndex = thisLen * maxD;
            
            // copyOut
            UpdateArgs updateArgs[MAX_ARGS_PIPE_LEN];
            for (int64_t i = 0; i < thisLen; i++) {
                int64_t thisIndForThisTable = uniqueIdGT.GetValue(offsetLen + i);
                int64_t embedDim = *(dOffsetsPtr + tableIndex + 1) - *(dOffsetsPtr + tableIndex);
                int64_t thisWeightOffset = *(weightsOffsetsPtr + tableIndex);
                int64_t thisOutOffset = thisWeightOffset + thisIndForThisTable * embedDim;
                updateArgs[i].embedDim = embedDim;
                updateArgs[i].thisOutOffset = thisOutOffset;
                DataCopy(inputLt[i * maxD + outIndex], momentum1DevGT[thisOutOffset], embedDim);
            }
            queIn.EnQue(inputLt);
            queOut.EnQue(outputLt);
            queIndices.EnQue(indicesLt);
            inputLt = queIn.DeQue<float>();
            outputLt = queOut.DeQue<float>();
            indicesLt = queIndices.DeQue<int64_t>();
            ComputeAda(inputLt, outputLt, thisLen);
            queOut.EnQue(outputLt);
            LocalTensor<float> newOutLt = queOut.DeQue<float>();
            SetAtomicAdd<float>();
            for (int64_t i = 0; i < thisLen; i++) {
                UpdateArgs theArgs = updateArgs[i];
                int64_t thisGradIndex = i * maxD;
                int64_t thisMomentIndex = i * maxD + outIndex;
                DataCopy(weightsDevOutGT[theArgs.thisOutOffset], newOutLt[thisGradIndex], theArgs.embedDim);
                DataCopy(momentum1DevOutGT[theArgs.thisOutOffset], newOutLt[thisMomentIndex], theArgs.embedDim);
            }
            SetAtomicNone();
            offsetLen += thisLen;
            thisLen = remain;
            queIn.FreeTensor(inputLt);
            queOut.FreeTensor(newOutLt);
            queIndices.FreeTensor(indicesLt);
        }
    }
    __aicore__ inline void Compute(Args args)
    {
        Init(args);
        InitUnique(args);
        ClearGrad();
        pipe_barrier(PIPE_ALL);
        SyncAll();
        ComputeGrad();
        pipe_barrier(PIPE_ALL);
        SyncAll();
        AdaScheduler();
    }
};
}  // namespace BackwardCodegenAdagradUnweightedExactUnique
#endif