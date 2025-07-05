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

#include "kernel_operator.h"

using namespace AscendC;

static constexpr int BURST_UNIT = 32;

template <typename T>
class SegmentSumY {
public:
    __aicore__ inline SegmentSumY() {}
    __aicore__ inline void Init(GM_ADDR dstGm, GM_ADDR srcGm, GM_ADDR segIndices, GM_ADDR tiling)
    {
        GET_TILING_DATA(tilingData, tiling);
        dimValue_ = tilingData.dimValue;
        batchSize_ = tilingData.batchSize;
        yDim_ = tilingData.yDim;
        zDim_ = tilingData.zDim;
        segNum_ = tilingData.segNum;
        srcGlobal_.SetGlobalBuffer((__gm__ T*)srcGm);
        dstGlobal_.SetGlobalBuffer((__gm__ T*)dstGm);
        segIndices_.SetGlobalBuffer((__gm__ uint32_t*)segIndices);

        pipe_.InitBuffer(outQueue_, 1, segNum_ * zDim_ * sizeof(T));
        pipe_.InitBuffer(inQueue_, 1, yDim_ * zDim_ * sizeof(T));

        const auto blockIdx = GetBlockIdx();
        const auto blockNum = GetBlockNum();
        if (blockNum == 0) {
            KERNEL_LOG(KERNEL_ERROR, "blockNum cannot be zero.");
            return;
        }
        const auto baseBatch = batchSize_ / blockNum;
        const auto remainder = batchSize_ % blockNum;

        if (blockIdx < remainder) {
            batchStart_ = blockIdx * (baseBatch + 1);
            batchEnd_ = batchStart_ + baseBatch + 1;
        } else {
            batchStart_ = remainder * (baseBatch + 1) + (blockIdx - remainder) * baseBatch;
            batchEnd_ = batchStart_ + baseBatch;
        }

        burstLen_ = zDim_ * sizeof(T) / BURST_UNIT;
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = batchStart_; i < batchEnd_; ++i) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(const uint32_t batchIdx)
    {
        SliceInfo srcSliceInfoIn[] = {{0, zDim_ - 1, 0, burstLen_, zDim_},
                                      {0, yDim_ - 1, 0, 1, yDim_},
                                      {batchIdx, batchIdx, 0, 1, batchSize_}};
        SliceInfo dstSliceInfoIn[] = {{0, zDim_ - 1, 0, burstLen_, zDim_},
                                      {0, yDim_ - 1, 0, 1, yDim_},
                                      {0, 0, 0, 1, 1}};
        LocalTensor<T> srcLocal = inQueue_.AllocTensor<T>();
        DataCopy(srcLocal, srcGlobal_, dstSliceInfoIn, srcSliceInfoIn, dimValue_);
        inQueue_.EnQue(srcLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<T> srcLocal = inQueue_.DeQue<T>();
        LocalTensor<T> outLocal = outQueue_.AllocTensor<T>();
        static constexpr CumSumConfig cumSumConfig{false, false, true};
        for (uint32_t i = 0; i < segNum_; ++i) {
            const uint32_t segStart = segIndices_.GetValue(i);
            const uint32_t segEnd = segIndices_.GetValue(i + 1);
            LocalTensor<T> srcSeg = srcLocal[segStart * zDim_];
            LocalTensor<T> outSeg = outLocal[i * zDim_];
            const CumSumInfo cumSumInfo{segEnd - segStart, zDim_};
            CumSum<T, cumSumConfig>(srcSeg, outSeg, srcSeg, cumSumInfo);
        }
        outQueue_.EnQue(outLocal);
        inQueue_.FreeTensor(srcLocal);
    }

    __aicore__ inline void CopyOut(const uint32_t batchIdx)
    {
        SliceInfo srcSliceInfoOut[] = {{0, zDim_ - 1, 0, burstLen_, zDim_},
                                       {0, segNum_ - 1, 0, 1, segNum_},
                                       {0, 0, 0, 1, 1}};
        SliceInfo dstSliceInfoOut[] = {{0, zDim_ - 1, 0, burstLen_, zDim_},
                                       {0, segNum_ - 1, 0, 1, segNum_},
                                       {batchIdx, batchIdx, 0, 1, batchSize_}};
        LocalTensor<T> outLocal = outQueue_.DeQue<T>();
        DataCopy(dstGlobal_, outLocal, dstSliceInfoOut, srcSliceInfoOut, dimValue_);
        outQueue_.FreeTensor(outLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    GlobalTensor<T> srcGlobal_;
    GlobalTensor<T> dstGlobal_;
    GlobalTensor<uint32_t> segIndices_;
    uint32_t dimValue_;

    uint32_t batchSize_;
    uint32_t segNum_;
    uint32_t yDim_;
    uint32_t zDim_;
    uint32_t burstLen_;
    uint32_t batchStart_;
    uint32_t batchEnd_;
};

extern "C" __global__ __aicore__ void segment_sum_y(GM_ADDR input, GM_ADDR segIndices, GM_ADDR output,
                                                    GM_ADDR workspace, GM_ADDR tiling)
{
    SegmentSumY<float> op;
    op.Init(output, input, segIndices, tiling);
    op.Process();
}
