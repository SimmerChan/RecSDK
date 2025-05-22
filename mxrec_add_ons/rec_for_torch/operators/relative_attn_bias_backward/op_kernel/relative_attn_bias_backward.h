/**
 * @file relative_attn_bias_backward.h
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 */

#ifndef MXREC_RELATIVE_ATTN_BIAS_BACKWARD_H
#define MXREC_RELATIVE_ATTN_BIAS_BACKWARD_H
#include "rab_common.h"
#include "kernel_operator.h"

template <typename FloatType>
class RelativeAttnBiasBackward {
public:
    __aicore__ inline RelativeAttnBiasBackward() {}

    __aicore__ inline void InitTensor(Args args)
    {
        tsGradGT.SetGlobalBuffer((__gm__ FloatType*)args.rabTimeGrad, numLayer * bs * s * s);
        bucketTimestampsGT.SetGlobalBuffer((__gm__ int32_t*)args.bucketTimestamps, numLayer * bs * s * s);
        tswGradOutGT.SetGlobalBuffer((__gm__ FloatType*)args.timestampsWeightsGrad, numLayer * bs * s * s);

        pipe.InitBuffer(inQueTsGrad, 1, AlignTo32(stride * sizeof(FloatType)));
        pipe.InitBuffer(inQueBucketTimestamps, 1, AlignTo32(stride * sizeof(int32_t)));
        pipe.InitBuffer(outQueTswGradOut, 1, AlignTo32(numBuckets * numLayer * sizeof(FloatType)));
    }

    __aicore__ inline void InitTiling()
    {
        int totalLen = bs * s * s;
        int totalTableSizeSplit = totalLen % GetBlockNum();
        int baseLen = totalLen / GetBlockNum();
        // 计算总共要处理的数据量、数据起始位置
        if (GetBlockIdx() >= totalTableSizeSplit) {
            processLen = baseLen;
            startGT = totalTableSizeSplit * (baseLen + 1) + (GetBlockIdx() - totalTableSizeSplit) * baseLen;
        } else {
            processLen = baseLen + 1;
            startGT = GetBlockIdx() * (baseLen + 1);
        }
    }

    __aicore__ inline void Init(Args args)
    {
        GET_TILING_DATA(tilingData, args.tiling);
        s = tilingData.s;
        bs = tilingData.bs;
        stride = tilingData.timeStride;
        numBuckets = tilingData.numBuckets;
        numLayer = tilingData.numLayer;

        InitTensor(args);
        InitTiling();
    }

    __aicore__ inline void InitTswGrad()
    {
        LocalTensor<FloatType> gradOut = outQueTswGradOut.AllocTensor<FloatType>();
        Duplicate(gradOut, (FloatType) 0, AlignTo32(numLayer * numBuckets * sizeof(FloatType)) / sizeof(FloatType));
        outQueTswGradOut.EnQue(gradOut);
    }

    __aicore__ inline void DataCopyInIndex(uint32_t offset, uint32_t cnt)
    {
        LocalTensor<int32_t> bucketTimestamps = inQueBucketTimestamps.AllocTensor<int32_t>();
        DataCopy(bucketTimestamps, bucketTimestampsGT[offset + startGT], cnt + DATA_ALIGN_BYTES / sizeof(int32_t));
        inQueBucketTimestamps.EnQue(bucketTimestamps);
    }

    __aicore__ inline void DataCopyInGrad(uint8_t layer, uint32_t offset, uint32_t cnt)
    {
        LocalTensor<FloatType> grad = inQueTsGrad.AllocTensor<FloatType>();
        DataCopy(grad, tsGradGT[offset + layer * bs * s * s], cnt + DATA_ALIGN_BYTES)
        inQueTsGrad.EnQue(grad);
    }

    __aicore__ inline void ScatterAdd(LocalTensor<FloatType> dst,
                                      LocalTensor<FloatType> src,
                                      LocalTensor<int32_t> index,
                                      uint8_t layer,
                                      uint32_t cnt)
    {
        __ubuf__ FloatType* dstAddr = reinterpret_cast<__ubuf__ FloatType*>(dst.GetPhyAddr());
        __ubuf__ FloatType* srcAddr = reinterpret_cast<__ubuf__ FloatType*>(src.GetPhyAddr());
        __ubuf__ int32_t* indexAddr = reinterpret_cast<__ubuf__ int32_t*>(index.GetPhyAddr());
        uint32_t layerOffset = layer * numBuckets;
        for (int i = 0; i < cnt; ++i) {
            const auto ind = indexAddr[i];
            const auto value = src[i];
            dst[layerOffset + ind] += value;
        }
    }

    __aicore__ inline void DataCopyOut(LocalTensor<FloatType> gradOut)
    {
        // 同步计算结果
        outQueTswGradOut.EnQue(gradOut);
        gradOut = outQueTswGradOut.DeQue<FloatType>;

        SetAtomicAdd<float>();
        DataCopy(tswGradOutGT, gradOut, AlignTo32(numLayer * numBuckets * sizeof(FloatType)) / sizeof(FloatType));
        SetAtomicNone();
    }

    __aicore__ inline void Compute(Args args)
    {
        Init(args);
        InitTswGrad();

        uint32_t offset = 0;
        LocalTensor<FloatType> gradOut = tswGradOutGT.DeQue<FloatType>();
        while (offset < processLen) {
            uint32_t remain = processLen - offset;
            uint32_t cnt = remain > stride ? stride : remain;

            DataCopyInIndex(offset, cnt);
            LocalTensor<int32_t> index = inQueBucketTimestamps.DeQue<int32_t>();
            for (uint8_t n = 0; n < numLayer; ++n) {
                DataCopyInGrad(n, offset, cnt);
                LocalTensor<FloatType> grad = tsGradGT.DeQue<FloatType>();
                ScatterAdd(gradOut, grad, index, n, cnt);
            }
            inQueBucketTimestamps.FreeTensor(index);
        }
        DataCopyOut(gradOut);
        outQueTswGradOut.FreeTensor(gradOut);
    }

private:
    GlobalTensor<FloatType> tsGradGT;
    GlobalTensor<int32_t> bucketTimestampsGT;
    GlobalTensor<FloatType> tswGradOutGT;

    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueTsGrad;
    TQue<TPosition::VECIN, 1> inQueBucketTimestamps;
    TQue<TPosition::VECOUT, 1> outQueTswGradOut;
private:
    // shape
    uint32_t s;
    uint32_t bs;
    uint32_t stride;
    uint32_t numBuckets;
    uint32_t numLayer;
    // tiling
    uint32_t processLen;
    uint32_t startGT;
};
#endif  // MXREC_RELATIVE_ATTN_BIAS_BACKWARD_H
