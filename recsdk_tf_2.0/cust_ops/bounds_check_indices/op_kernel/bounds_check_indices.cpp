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

#include "kernel_operator.h"

using namespace AscendC;

namespace BoundsCheckIndices {
constexpr int64_t CHECK_MODE_FATAL = 0;
constexpr int64_t CHECK_MODE_WARNING = 1;
constexpr int64_t CHECK_MODE_IGNORE = 2;

constexpr uint32_t DATA_ALIGN_BYTES = 32;

template <typename T>
__aicore__ inline void CpGm2Local(const LocalTensor<T>& lt, const GlobalTensor<T>& gt, int64_t len)
{
    uint32_t alignLen = len * sizeof(T) / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES;
    uint32_t unAlignLen = len * sizeof(T) - alignLen;

    GlobalTensor<uint8_t> uint8Gt;
    uint8Gt.SetGlobalBuffer((__gm__ uint8_t*)gt.GetPhyAddr(), len * sizeof(T));
    LocalTensor<uint8_t> uint8Lt = lt.template ReinterpretCast<uint8_t>();

    if (alignLen != 0) {
        DataCopy(uint8Lt, uint8Gt, alignLen);
    }
    if (unAlignLen != 0) {
        const DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
        const DataCopyPadExtParams<uint8_t> dataCopyPadExtParams{false, 0, 0, 0};
        DataCopyPad(uint8Lt[alignLen], uint8Gt[alignLen], dataCopyExtParams, dataCopyPadExtParams);
    }
}

template <typename T>
__aicore__ inline void CpLocal2Gm(const GlobalTensor<T>& gt, const LocalTensor<T>& lt, int64_t len)
{
    uint32_t alignLen = len * sizeof(T) / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES;
    uint32_t unAlignLen = len * sizeof(T) - alignLen;

    GlobalTensor<uint8_t> uint8Gt;
    uint8Gt.SetGlobalBuffer((__gm__ uint8_t*)gt.GetPhyAddr(), len * sizeof(T));
    LocalTensor<uint8_t> uint8Lt = lt.template ReinterpretCast<uint8_t>();

    if (alignLen != 0) {
        DataCopy(uint8Gt, uint8Lt, alignLen);
    }
    if (unAlignLen != 0) {
        const DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
        DataCopyPad(uint8Gt[alignLen], uint8Lt[alignLen], dataCopyExtParams);
    }
}

struct CheckKernelArgs {
    GM_ADDR rowsPtr;
    GM_ADDR indicesPtr;
    GM_ADDR offsetsPtr;
    GM_ADDR warning;
    GM_ADDR tiling;
    GM_ADDR workspace;
};

class BoundsCheckIndicesKenel {
public:
    __aicore__ inline BoundsCheckIndicesKenel(const CheckKernelArgs& args)
    {
        GET_TILING_DATA(tilingData, args.tiling);
        numTable_ = tilingData.numTable;
        numIndices_ = tilingData.numIndices;
        totalNumBatch_ = tilingData.totalNumBatch;
        numBatch_ = tilingData.numBatch;
        boundsCheckMode_ = tilingData.boundsCheckMode;
        coresPerTable_ = tilingData.coresPerTable;

        rowsPtr_ = args.rowsPtr;
        indicesPtr_ = args.indicesPtr;
        offsetsPtr_ = args.offsetsPtr;
        warning_ = args.warning;
        workspace_ = args.workspace;
        blockNum_ = GetBlockNum();
        blockIdx_ = GetBlockIdx();
        tableIdx_ = blockIdx_ / coresPerTable_;
        numRows_ = *(reinterpret_cast<__gm__ int64_t*>(rowsPtr_) + tableIdx_);

        indicesGT_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(indicesPtr_), numIndices_ * sizeof(int64_t));
        offsetsGT_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(offsetsPtr_),
                                   (totalNumBatch_ + 1) * sizeof(int64_t));
        warningGT_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(warning_), numTable_ * sizeof(int64_t));

        // There are some black magic behind `InitBuffer`, reorder will cause `Ascend` crash.
        pipe_.InitBuffer(warningQue_, 1, sizeof(int64_t));
        pipe_.InitBuffer(offsetsQue_, 1, (numBatch_ + 1) * sizeof(int64_t));
        pipe_.InitBuffer(indicesQue_, 1, numIndices_ * sizeof(int64_t));
    }

    __aicore__ inline void Compute()
    {
        if (numBatch_ * numTable_ != totalNumBatch_) {
            return;
        }

        CheckKernel();

        // check offsets[totalNumBatch]
        if (blockIdx_ == 0) {
            LocalTensor<uint8_t> offsetsLt = offsetsQue_.AllocTensor<uint8_t>();
            CpGm2Local(offsetsLt, offsetsGT_[totalNumBatch_ * sizeof(int64_t)], sizeof(int64_t));
            LocalTensor<int64_t> offsetsLt64 = offsetsLt.template ReinterpretCast<int64_t>();
            const int64_t offsetsLast = static_cast<int64_t>(offsetsLt64.GetValue(0));
            if (offsetsLast != numIndices_) {
                if (boundsCheckMode_ == CHECK_MODE_FATAL) {
                    ASCENDC_ASSERT(false,
                                   { KERNEL_LOG(KERNEL_ERROR, "CheckMode is Fatal, and some data is out of bounds"); });
                    return;
                } else if (boundsCheckMode_ == CHECK_MODE_WARNING) {
                    localWarning_ += 1;
                    offsetsLt64.SetValue(0, numIndices_);
                } else if (boundsCheckMode_ == CHECK_MODE_IGNORE) {
                    offsetsLt64.SetValue(0, numIndices_);
                }
            }
            offsetsQue_.EnQue(offsetsLt);
            LocalTensor<uint8_t> outOffsetsLt = offsetsQue_.DeQue<uint8_t>();
            CpLocal2Gm(offsetsGT_[totalNumBatch_ * sizeof(int64_t)], outOffsetsLt, sizeof(int64_t));
            offsetsQue_.FreeTensor(outOffsetsLt);
        }

        LocalTensor<uint8_t> warningLt = warningQue_.AllocTensor<uint8_t>();
        CpGm2Local(warningLt, warningGT_[blockIdx_ * sizeof(int64_t)], sizeof(int64_t));
        LocalTensor<int64_t> warningLt64 = warningLt.template ReinterpretCast<int64_t>();
        warningLt64.SetValue(0, static_cast<int64_t>(localWarning_));
        warningQue_.EnQue(warningLt);
        LocalTensor<uint8_t> outWarningLt = warningQue_.DeQue<uint8_t>();
        CpLocal2Gm(warningGT_[blockIdx_ * sizeof(int64_t)], outWarningLt, sizeof(int64_t));
        warningQue_.FreeTensor(outWarningLt);
    }

private:
    __aicore__ inline void checkIndices(LocalTensor<int64_t>& indicesLt64, int64_t indicesRange)
    {
        for (int64_t l = 0; l < indicesRange; ++l) {
            const auto idx = indicesLt64.GetValue(l);
            if (idx == -1) {
                // -1 indicates pruned rows.
                continue;
            }
            if (idx < 0 || idx >= numRows_) {
                if (boundsCheckMode_ == CHECK_MODE_FATAL) {
                    ASCENDC_ASSERT(false,
                                   { KERNEL_LOG(KERNEL_ERROR, "CheckMode is Fatal, and some data is out of bounds"); });
                    return;
                } else if (boundsCheckMode_ == CHECK_MODE_WARNING) {
                    localWarning_ += 1;
                    indicesLt64.SetValue(l, 0);
                } else if (boundsCheckMode_ == CHECK_MODE_IGNORE) {
                    indicesLt64.SetValue(l, 0);
                }
            }
        }
    }

    __aicore__ inline void checkOffsets(LocalTensor<int64_t>& offsetsLt64, int64_t batchRange)
    {
        for (int64_t i = 0; i < batchRange; ++i) {
            int64_t indicesStart = static_cast<int64_t>(offsetsLt64.GetValue(i));
            int64_t indicesEnd = static_cast<int64_t>(offsetsLt64.GetValue(i + 1));
            if (indicesStart < 0 || indicesStart > indicesEnd || indicesEnd > numIndices_) {
                if (boundsCheckMode_ == CHECK_MODE_FATAL) {
                    ASCENDC_ASSERT(false,
                                   { KERNEL_LOG(KERNEL_ERROR, "CheckMode is Fatal, and some data is out of bounds"); });
                    return;
                } else if (boundsCheckMode_ == CHECK_MODE_WARNING) {
                    localWarning_ += 1;
                    indicesStart = max(0L, min(indicesStart, numIndices_));
                    indicesEnd = max(indicesStart, min(indicesEnd, numIndices_));
                    offsetsLt64.SetValue(i, indicesStart);
                    offsetsLt64.SetValue(i + 1, indicesEnd);
                } else if (boundsCheckMode_ == CHECK_MODE_IGNORE) {
                    indicesStart = max(0L, min(indicesStart, numIndices_));
                    indicesEnd = max(indicesStart, min(indicesEnd, numIndices_));
                    offsetsLt64.SetValue(i, indicesStart);
                    offsetsLt64.SetValue(i + 1, indicesEnd);
                }
            }
            const auto indicesRange = indicesEnd - indicesStart;
            LocalTensor<uint8_t> indicesLt = indicesQue_.AllocTensor<uint8_t>();
            CpGm2Local(indicesLt, indicesGT_[indicesStart * sizeof(int64_t)], indicesRange * sizeof(int64_t));
            LocalTensor<int64_t> indicesLt64 = indicesLt.template ReinterpretCast<int64_t>();
            checkIndices(indicesLt64, indicesRange);
            indicesQue_.EnQue(indicesLt);
            LocalTensor<uint8_t> outIndicesLt = indicesQue_.DeQue<uint8_t>();
            CpLocal2Gm(indicesGT_[indicesStart * sizeof(int64_t)], outIndicesLt, indicesRange * sizeof(int64_t));
            indicesQue_.FreeTensor(outIndicesLt);
        }
    }

    __aicore__ inline void CheckKernel()
    {
        const int64_t batchBegin = tableIdx_ * numBatch_;
        const int64_t batchEnd = (tableIdx_ + 1) * numBatch_;
        const int64_t miniStep = numBatch_ / coresPerTable_;
        const int64_t miniIdx = blockIdx_ % coresPerTable_;
        const int64_t miniBatchBegin = batchBegin + miniIdx * miniStep;
        const int64_t miniBatchEnd = miniIdx == coresPerTable_ - 1 ? batchEnd : (miniBatchBegin + miniStep);
        const auto batchRange = miniBatchEnd - miniBatchBegin;
        LocalTensor<uint8_t> offsetsLt = offsetsQue_.AllocTensor<uint8_t>();
        CpGm2Local(offsetsLt, offsetsGT_[miniBatchBegin * sizeof(int64_t)], (batchRange + 1) * sizeof(int64_t));
        LocalTensor<int64_t> offsetsLt64 = offsetsLt.template ReinterpretCast<int64_t>();
        checkOffsets(offsetsLt64, batchRange);
        offsetsQue_.EnQue(offsetsLt);
        LocalTensor<uint8_t> outOffsetsLt = offsetsQue_.DeQue<uint8_t>();
        CpLocal2Gm(offsetsGT_[miniBatchBegin * sizeof(int64_t)], outOffsetsLt, (batchRange + 1) * sizeof(int64_t));
        offsetsQue_.FreeTensor(outOffsetsLt);
    }

    int64_t blockNum_;
    int64_t blockIdx_;
    int64_t tableIdx_;
    int64_t numRows_;

    // tiling
    int64_t numTable_;
    int64_t numIndices_;
    int64_t totalNumBatch_;
    int64_t numBatch_;
    int64_t boundsCheckMode_;
    int64_t coresPerTable_;

    // raw global memory pointers
    GM_ADDR rowsPtr_;
    GM_ADDR indicesPtr_;
    GM_ADDR offsetsPtr_;
    GM_ADDR warning_;
    GM_ADDR workspace_;

    // pipe
    TPipe pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> indicesQue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> offsetsQue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> warningQue_;

    // global tensors
    GlobalTensor<uint8_t> indicesGT_;
    GlobalTensor<uint8_t> offsetsGT_;
    GlobalTensor<uint8_t> warningGT_;

    int32_t localWarning_{0};
};
}  // namespace BoundsCheckIndices

extern "C" __global__ __aicore__ void bounds_check_indices(GM_ADDR rows_per_table, GM_ADDR indices, GM_ADDR offsets,
                                                           GM_ADDR warning, GM_ADDR workspace, GM_ADDR tiling)
{
    BoundsCheckIndices::CheckKernelArgs args{rows_per_table, indices, offsets, warning, tiling, workspace};
    BoundsCheckIndices::BoundsCheckIndicesKenel opKernel(args);
    opKernel.Compute();
}
