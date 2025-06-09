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
namespace BackwardCodegenAdagradUnweightedExact {

constexpr int NUM_OF_OUT = 2;
template <typename wType>
class BackwardCodegenAdagradUnweightedExactKernelUnique : public BackwardCodegenUnweightedExactKernelUnique<wType> {
public:
    __aicore__ inline BackwardCodegenAdagradUnweightedExactKernelUnique() {}
    __aicore__ inline void AdaScheduler()
    {
        int64_t lastIndices = 0;
        for (int64_t i = 1; i < this->this->uniqueHashDim0; i++) {
            if (uniqueHashSizeGT.GetValue(i) != lastIndices) {
                Scheduler(uniqueHashSizeGT.GetValue(i) - lastIndices, this->offsetOfThisCore, this->lenOfThisCore);
                if (this->lenOfThisCore > 0) {
                    UpdateEmbedAda(this->offsetOfThisCore + lastIndices, this->lenOfThisCore, i - 1);
                }
                lastIndices = uniqueHashSizeGT.GetValue(i);
            }
        }
    }

    __aicore__ inline void ComputeAda(LocalTensor<float>inputLt, LocalTensor<float>outLt, int64_t thisLen)
    {
        int64_t momentum1Offset = thisLen * this->maxD;
        Mul<float>(outLt, inputLt, inputLt, momentum1Offset);
        Add<float>(outLt, inputLt[momentum1Offset], outLt, momentum1Offset);
        Sqrt<float>(outLt, outLt, momentum1Offset);
        Adds<float>(outLt, outLt, this->eps, momentum1Offset);
        Duplicate<float>(outLt[momentum1Offset], this->learning_rate, momentum1Offset);
        Div<float>(outLt, outLt[momentum1Offset], outLt, momentum1Offset);
        Mul<float>(outLt, outLt, inputLt, momentum1Offset);
        Muls<float>(outLt, outLt, -1, momentum1Offset);
        Mul<float>(outLt[momentum1Offset], inputLt, inputLt, momentum1Offset);
    }

    __aicore__ inline void UpdateEmbedAda(int64_t offsetLen, int64_t totalLen, int64_t tableIndex)
    {
        __gm__ int32_t* dOffsetsPtr = (__gm__ int32_t*)this->dOffsets;
        __gm__ int64_t* weightsOffsetsPtr = (__gm__ int64_t*)this->weightsOffsets;
        int indicesNumOneBlock = this->blockLen / NUM_OF_OUT / this->maxD;
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
            LocalTensor<float> inputLt = this->queIn.template AllocTensor<float>();
            LocalTensor<float> outputLt = this->queOut.template AllocTensor<float>();
            LocalTensor<int64_t> indicesLt = this->queIndices.template AllocTensor<int64_t>();
            
            // copyIn
            CpGm2Local(indicesLt, this->uniqueIdGT[offsetLen], thisLen);
            CpGm2Local(inputLt, this->outGT[offsetLen * this->maxD], thisLen * this->maxD);
            int64_t outIndex = thisLen * this->maxD;
            
            // copyOut
            UpdateArgs updateArgs[MAX_ARGS_PIPE_LEN];
            for (int64_t i = 0; i < thisLen; i++) {
                int64_t thisIndForThisTable = this->uniqueIdGT.GetValue(offsetLen + i);
                // int64_t m1Addr = this->momentum1DevGT.GetValue(offsetLen + i);
                int64_t embedDim = *(dOffsetsPtr + tableIndex + 1) - *(dOffsetsPtr + tableIndex);
                int64_t thisWeightOffset = *(weightsOffsetsPtr + tableIndex);
                int64_t thisOutOffset = thisWeightOffset + thisIndForThisTable * embedDim;
                updateArgs[i].embedDim = embedDim;
                updateArgs[i].thisOutOffset = thisOutOffset;
                DataCopy(inputLt[i * this->maxD + outIndex], this->momentum1DevGT[thisOutOffset], embedDim);
            }
            this->queIn.template EnQue(inputLt);
            this->queOut.template EnQue(outputLt);
            this->queIndices.template EnQue(indicesLt);
            inputLt = this->queIn.template DeQue<float>();
            outputLt = this->queOut.template DeQue<float>();
            indicesLt = this->queIndices.template DeQue<int64_t>();
            ComputeAda(inputLt, outputLt, thisLen);
            this->queOut.template EnQue(outputLt);
            LocalTensor<float> newOutLt = this->queOut.template DeQue<float>();
            SetAtomicAdd<float>();
            for (int64_t i = 0; i < thisLen; i++) {
                UpdateArgs theArgs = updateArgs[i];
                int64_t thisGradIndex = i * this->maxD;
                int64_t thisMomentIndex = i * this->maxD + outIndex;
                DataCopy(this->weightsDevOutGT[theArgs.thisOutOffset], newOutLt[thisGradIndex], theArgs.embedDim);
                DataCopy(this->momentum1DevOutGT[theArgs.thisOutOffset], newOutLt[thisMomentIndex], theArgs.embedDim);
            }
            SetAtomicNone();
            offsetLen += thisLen;
            thisLen = remain;
            this->queIn.template FreeTensor(inputLt);
            this->queOut.template FreeTensor(newOutLt);
            this->queIndices.template FreeTensor(indicesLt);
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