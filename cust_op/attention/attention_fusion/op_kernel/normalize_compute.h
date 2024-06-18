#ifndef NORMALIZE_COMPUTE__H
#define NORMALIZE_COMPUTE__H
#include <cstdint>
#include "kernel_operator.h"

using namespace AscendC;

struct NormalizeArgs {
    TPipe* pipe;

    uint8_t attr;
    int queryDim1;
    int keyDim1;
    int batchOffset;
    int batchLen;
    int loopCount;
    int normalizeRow;
    int normalizeColumn;
    float normalizeSqrt;
    uint64_t maxSharedTmpBuf;

    const SoftMaxTiling* tiling;

    const ConfusionTransposeTiling* confusionTransposeTilingData;
    const ConfusionTransposeTiling* confusionTransposeTilingData1;
    const ConfusionTransposeTiling* confusionTransposeTilingData2;
    const ConfusionTransposeTiling* confusionTransposeTilingData3;
};

template<typename qType>
class NormalizeCompute {
public:
    __aicore__ inline NormalizeCompute(){}

    __aicore__ inline void Init(NormalizeArgs normalArgs)
    {
        this->args = normalArgs;
        int bufSize = args.normalizeRow * args.normalizeColumn * sizeof(qType);
        args.pipe->InitBuffer(vecInQueue, 1, bufSize);
        args.pipe->InitBuffer(vecOutQueue, 1, bufSize);
        args.pipe->InitBuffer(vecSharedQueue, 1, args.maxSharedTmpBuf);
    }

    __aicore__ inline void DoPadLocal(LocalTensor<qType>& sourceTensor, LocalTensor<qType>& mindTensor,
                                        const ConfusionTransposeTiling* confusionTransposeTilingData,
                                        const ConfusionTransposeTiling* confusionTransposeTilingData1)
    {
        int totalSize = 16 * 50 * 16;
        int padSize = 16 * 56 * 16;

        ConfusionTransposeTiling tiling = *confusionTransposeTilingData;
        ConfusionTranspose<qType>(mindTensor, sourceTensor, TransposeType::TRANSPOSE_ND2ND_ONLY, tiling);
        Duplicate<float>(sourceTensor, -1000, padSize);
        DataCopyParams dataCopyParam = {0, 0, 0, 0};
        dataCopyParam.blockCount = 16;
        dataCopyParam.blockLen = 50 *16 / 8;
        dataCopyParam.srcStride = 0;
        dataCopyParam.dstStride = 6 * 16 / 8;
        DataCopy(sourceTensor, mindTensor, dataCopyParam);

        ConfusionTransposeTiling tiling1 = *confusionTransposeTilingData1;
        ConfusionTranspose<qType>(mindTensor, sourceTensor, TransposeType::TRANSPOSE_ND2ND_ONLY, tiling1);
        DataCopy(sourceTensor, mindTensor, padSize);
    }


    __aicore__ inline void DoUnPadLocal(LocalTensor<qType>& sourceTensor, LocalTensor<qType>& mindTensor,
                                        const ConfusionTransposeTiling* confusionTransposeTilingData2,
                                        const ConfusionTransposeTiling* confusionTransposeTilingData3)
    {
        int totalSize = 16 * 50 * 16;
        int padSize = 16 * 56 * 16;

        ConfusionTransposeTiling tiling = *confusionTransposeTilingData2;
        ConfusionTranspose<qType>(mindTensor, sourceTensor, TransposeType::TRANSPOSE_ND2ND_ONLY, tiling);
        Duplicate<float>(sourceTensor, -1000, padSize);
        DataCopyParams dataCopyParam = {0, 0, 0, 0};
        dataCopyParam.blockCount = 16;
        dataCopyParam.blockLen = 50 *16 / 8;
        dataCopyParam.srcStride = 6 * 16 / 8;
        dataCopyParam.dstStride = 0;
        DataCopy(sourceTensor, mindTensor, dataCopyParam);

        ConfusionTransposeTiling tiling1 = *confusionTransposeTilingData3;
        ConfusionTranspose<qType>(mindTensor, sourceTensor, TransposeType::TRANSPOSE_ND2ND_ONLY, tiling1);
        DataCopy(sourceTensor, mindTensor, padSize);
    }
    
    __aicore__ inline void Process(GlobalTensor<qType> softmaxGlobleTensor, GlobalTensor<qType> softmaxGbMask)
    {
        srcGloblePtr = softmaxGlobleTensor;
        maskGloblePtr = softmaxGbMask;
        offset = 0;
        usedRowCount = 0;
        uint8_t padLen = args.normalizeColumn - args.keyDim1;
        padParams = {false, 0, padLen, 0};

        for (int i = 0; i < args.loopCount; i++) {
            /* Get height of softmax matrix and handle the last loop height */
            height = ((args.queryDim1 - usedRowCount) < args.normalizeRow) ?
                        args.queryDim1 - usedRowCount : args.normalizeRow;
            totalSize = height * args.normalizeColumn;
            CopyIn();
            Compute();
            CopyOut();

            usedRowCount += height;
            offset += args.normalizeRow * args.keyDim1;
        }
    }

private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<qType> inLocalTensor = vecInQueue.AllocTensor<qType>();
        LocalTensor<qType> LocalMask = vecSharedQueue.AllocTensor<qType>();
        
        if (args.attr == 1) {
            DataCopy(inLocalTensor, srcGloblePtr[offset], totalSize);
            DataCopy(LocalMask, maskGloblePtr[offset], totalSize);
        } else if (args.attr == 2) {
            DataCopy(inLocalTensor, srcGloblePtr[offset], totalSize);
            DataCopy(LocalMask, maskGloblePtr[offset], totalSize);     
        } else {
            copyParams.blockCount = height;
            copyParams.blockLen = args.keyDim1 * sizeof(qType);
            DataCopyPad(inLocalTensor, srcGloblePtr[offset], copyParams, padParams);
            DataCopyPad(LocalMask, maskGloblePtr[offset], copyParams, padParams);
        }

        vecInQueue.EnQue(inLocalTensor);
        vecSharedQueue.EnQue(LocalMask);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<qType> inLocalTensor = vecInQueue.DeQue<qType>();
        LocalTensor<qType> LocalMask = vecSharedQueue.DeQue<qType>();
        LocalTensor<qType> outLocalTensor = vecOutQueue.AllocTensor<qType>();

        if (args.attr == 2) {
            DoPadLocal(LocalMask, outLocalTensor, args.confusionTransposeTilingData,
                                                    args.confusionTransposeTilingData1);
            DoPadLocal(inLocalTensor, outLocalTensor, args.confusionTransposeTilingData,
                                                        args.confusionTransposeTilingData1);
        }

        // atten_weight = qkMatMul / sqrt(atten_dim)
        Muls(outLocalTensor, inLocalTensor, args.normalizeSqrt, totalSize);

        // atten_mask = (1 - mask) * 10000
        Muls(LocalMask, LocalMask, (float)-10000, totalSize);
        Adds(LocalMask, LocalMask, (float)10000, totalSize);

        // atten_weight = atten_weight + atten_mask
        Add(inLocalTensor, outLocalTensor, LocalMask, totalSize);
        vecSharedQueue.FreeTensor(LocalMask);

        LocalTensor<uint8_t> sharedTmpBuf = vecSharedQueue.AllocTensor<uint8_t>();

        SoftMaxShapeInfo scrShape ={height, (uint32_t)args.normalizeColumn, height, (uint32_t)args.keyDim1};
        SoftMax<qType>(outLocalTensor, inLocalTensor, sharedTmpBuf, *args.tiling, scrShape);

        if (args.attr == 2) {
            DoUnPadLocal(outLocalTensor, inLocalTensor, args.confusionTransposeTilingData2,
                                                        args.confusionTransposeTilingData3);
        }
        vecOutQueue.EnQue<qType>(outLocalTensor);
        vecInQueue.FreeTensor(inLocalTensor);
        vecSharedQueue.FreeTensor(sharedTmpBuf);
    }

    __aicore__ inline void CopyOut()
    {
        LocalTensor<qType> outLocalTensor = vecOutQueue.DeQue<qType>();

        if (args.attr == 1) {
            DataCopy(srcGloblePtr[offset], outLocalTensor, totalSize);
        } else if (args.attr == 2) {
            uint32_t thisLen = height * args.keyDim1 * sizeof(qType);
            if (thisLen % 32 != 0) {
                DataCopyExtParams dataCopyParamTail {1, thisLen, 0, 0, 0};
                DataCopyPad(srcGloblePtr[offset], outLocalTensor, dataCopyParamTail);
            } else {
                DataCopy(srcGloblePtr[offset], outLocalTensor, height * args.keyDim1);
            }
        } else {
            DataCopyPad(srcGloblePtr[offset], outLocalTensor, copyParams);
        }
        vecOutQueue.FreeTensor(outLocalTensor);
    }

private:
    NormalizeArgs args;
    TQue<QuePosition::VECIN, 1> vecInQueue;
    TQue<QuePosition::VECOUT, 1> vecOutQueue;
    TQue<QuePosition::VECIN, 1> vecSharedQueue;

    GlobalTensor<qType> srcGloblePtr;
    GlobalTensor<qType> maskGloblePtr;
    uint32_t height = 0;
    int offset = 0;
    int usedRowCount = 0;
    uint32_t totalSize = 0;
    struct DataCopyExtParams copyParams;
    struct DataCopyPadExtParams<qType> padParams;
};
#endif