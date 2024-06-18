#ifndef NORMALIZE_GRAD_COMPUTE_H
#define NORMALIZE_GRAD_COMPUTE_H
#include <cstdint>
#include "kernel_operator.h"
#include "utils.h"
using namespace AscendC;

struct NormGradArgs {
    GM_ADDR softmaxOut;
    GM_ADDR gradSoftmax;

    int sDim1;
    int sDim2;

    int batchNum;
    int batchOffset;
    int batchLen;

    int numOfNormalnizeOnce;
    int paddingKeyDim1;
    float attenDimSqrt;
    int keyDimAlign;

    const SoftMaxTiling* softmaxtiling;
    const ConfusionTransposeTiling* confusionTransposeTilingData;
    const ConfusionTransposeTiling* confusionTransposeTilingData1;
    const ConfusionTransposeTiling* confusionTransposeTilingData2;
    const ConfusionTransposeTiling* confusionTransposeTilingData3;
};

struct NormGradPipeArgs {
    TPipe* pipe;
};

template<typename tType>
class NormalGradCompute {
public:
    __aicore__ inline NormalGradCompute(){}

    __aicore__ inline void Init(NormGradArgs mmArgs, NormGradPipeArgs pipeArgs){
        this->mmArgs = mmArgs;
        softmaxOut.SetGlobalBuffer(reinterpret_cast<__gm__ tType*>(mmArgs.softmaxOut), mmArgs.batchNum * mmArgs.sDim1 * mmArgs.sDim2);
        softmaxOut = softmaxOut[mmArgs.batchOffset * mmArgs.sDim1 * mmArgs.sDim2];

        gradSoftmax.SetGlobalBuffer(reinterpret_cast<__gm__ tType*>(mmArgs.gradSoftmax), mmArgs.batchNum * mmArgs.sDim1 * mmArgs.sDim2);
        gradSoftmax = gradSoftmax[mmArgs.batchOffset * mmArgs.sDim1 * mmArgs.sDim2];

        pipeArgs.pipe->InitBuffer(vecInQueue, 1, mmArgs.numOfNormalnizeOnce*mmArgs.paddingKeyDim1*sizeof(tType));
        pipeArgs.pipe->InitBuffer(vecInGradQueue, 1, mmArgs.numOfNormalnizeOnce*mmArgs.paddingKeyDim1*sizeof(tType));
        pipeArgs.pipe->InitBuffer(vecOutQueue, 1, mmArgs.numOfNormalnizeOnce*mmArgs.paddingKeyDim1*sizeof(tType));
        pipeArgs.pipe->InitBuffer(tmpBuff,  mmArgs.numOfNormalnizeOnce*mmArgs.paddingKeyDim1*sizeof(tType));
    }

    __aicore__ inline void DoPadLocal(LocalTensor<tType>& sourceTensor, LocalTensor<tType>& mindTensor, const ConfusionTransposeTiling* confusionTransposeTilingData,  const ConfusionTransposeTiling* confusionTransposeTilingData1)
    {

        int totalSize = 16 * 50 * 8;
        int padSize = 16 * 56 * 8;

        ConfusionTransposeTiling tiling = *confusionTransposeTilingData;
        ConfusionTranspose<tType>(mindTensor, sourceTensor, TransposeType::TRANSPOSE_ND2ND_ONLY, tiling);
        Duplicate<float>(sourceTensor, 0, padSize);
        DataCopyParams dataCopyParam = {0, 0, 0, 0};
        dataCopyParam.blockCount = 8;
        dataCopyParam.blockLen = 50 *16 / 8;
        dataCopyParam.srcStride = 0;
        dataCopyParam.dstStride = 6 * 16 / 8;
        DataCopy(sourceTensor, mindTensor, dataCopyParam);

        ConfusionTransposeTiling tiling1 = *confusionTransposeTilingData1;
        ConfusionTranspose<tType>(mindTensor, sourceTensor, TransposeType::TRANSPOSE_ND2ND_ONLY, tiling1);
        DataCopy(sourceTensor, mindTensor, padSize);
    }


    __aicore__ inline void DoUnPadLocal(LocalTensor<tType>& sourceTensor, LocalTensor<tType>& mindTensor, const ConfusionTransposeTiling* confusionTransposeTilingData2,  const ConfusionTransposeTiling* confusionTransposeTilingData3)
    {

        int totalSize = 16 * 50 * 8;
        int padSize = 16 * 56 * 8;

        ConfusionTransposeTiling tiling = *confusionTransposeTilingData2;
        ConfusionTranspose<tType>(mindTensor, sourceTensor, TransposeType::TRANSPOSE_ND2ND_ONLY, tiling);
        Duplicate<float>(sourceTensor, 0, padSize);
        DataCopyParams dataCopyParam = {0, 0, 0, 0};
        dataCopyParam.blockCount = 8;
        dataCopyParam.blockLen = 50 *16 / 8;
        dataCopyParam.srcStride = 6 * 16 / 8;
        dataCopyParam.dstStride = 0;
        DataCopy(sourceTensor, mindTensor, dataCopyParam);

        ConfusionTransposeTiling tiling1 = *confusionTransposeTilingData3;
        ConfusionTranspose<tType>(mindTensor, sourceTensor, TransposeType::TRANSPOSE_ND2ND_ONLY, tiling1);
        DataCopy(sourceTensor, mindTensor, padSize);

    }

    __aicore__ inline void ProcessOneBatch(uint32_t batchI){

        struct DataCopyExtParams copyParams{0, 0, 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
        // struct DataCopyPadExtParams<float> padParams{false, 0, 0, 0}; 
        DataCopyPadExtParams<tType> padParams{true, 0, (uint8_t) (mmArgs.paddingKeyDim1-mmArgs.sDim2), 0};
        GlobalTensor<tType> thisBatchSoftmaxGb = softmaxOut[batchI * mmArgs.sDim1 * mmArgs.sDim2];
        GlobalTensor<tType> thisBatchSoftmaxGradGb = gradSoftmax[batchI * mmArgs.sDim1 * mmArgs.sDim2];

        int total = mmArgs.sDim1 * mmArgs.sDim2;
        int remain = total;

        while (remain > 0) {
            // caculate basic
            int thisLen = mmArgs.numOfNormalnizeOnce * mmArgs.sDim2;
            if (remain < thisLen) {
                thisLen = remain;
            }
            int offset = total - remain;

            copyParams.blockCount = thisLen/mmArgs.sDim2;
            copyParams.blockLen = mmArgs.sDim2 * sizeof(tType);

            LocalTensor<tType> inLocalTensor = vecInQueue.AllocTensor<tType>();
            LocalTensor<tType> inGradLocalTensor = vecInGradQueue.AllocTensor<tType>();
            if (mmArgs.keyDimAlign == 1 || mmArgs.keyDimAlign == 2) {
                DataCopy(inGradLocalTensor, thisBatchSoftmaxGradGb[offset], thisLen / mmArgs.sDim2 * mmArgs.paddingKeyDim1);
                DataCopy(inLocalTensor, thisBatchSoftmaxGb[offset], thisLen / mmArgs.sDim2 * mmArgs.paddingKeyDim1);
            } else {
                DataCopyPad(inGradLocalTensor, thisBatchSoftmaxGradGb[offset], copyParams, padParams);
                DataCopyPad(inLocalTensor, thisBatchSoftmaxGb[offset], copyParams, padParams);
            }
            vecInQueue.EnQue(inLocalTensor);
            vecInGradQueue.EnQue(inGradLocalTensor);

            LocalTensor<tType> inLocalTensorCompute = vecInQueue.DeQue<tType>();
            LocalTensor<tType> inGradLocalTensorCompute = vecInGradQueue.DeQue<tType>();
            LocalTensor<tType> outLocalTensor = vecOutQueue.AllocTensor<tType>();

            if (mmArgs.keyDimAlign == 2) {
                DoPadLocal(inLocalTensorCompute, outLocalTensor, mmArgs.confusionTransposeTilingData, mmArgs.confusionTransposeTilingData1);
                DoPadLocal(inGradLocalTensorCompute, outLocalTensor, mmArgs.confusionTransposeTilingData, mmArgs.confusionTransposeTilingData1);
            }


            SoftMaxShapeInfo scrShape ={(uint32_t)thisLen/mmArgs.sDim2, (uint32_t)mmArgs.paddingKeyDim1, (uint32_t)thisLen/mmArgs.sDim2, (uint32_t) mmArgs.sDim2};

            SoftmaxGrad<tType>(outLocalTensor,  inGradLocalTensorCompute , inLocalTensorCompute, tmpBuff.Get<uint8_t>(),*mmArgs.softmaxtiling, false, scrShape);
            Muls(outLocalTensor, outLocalTensor, mmArgs.attenDimSqrt, thisLen/mmArgs.sDim2*mmArgs.paddingKeyDim1);

            if (mmArgs.keyDimAlign == 2) {
                DoUnPadLocal(outLocalTensor, inLocalTensorCompute, mmArgs.confusionTransposeTilingData2, mmArgs.confusionTransposeTilingData3);
            }
            vecInQueue.FreeTensor(inLocalTensorCompute);
            vecInGradQueue.FreeTensor(inGradLocalTensorCompute);

            vecOutQueue.EnQue<tType>(outLocalTensor);

            LocalTensor<tType> copyOutLocalTensor = vecOutQueue.DeQue<tType>();

            if (mmArgs.keyDimAlign == 1 || mmArgs.keyDimAlign == 2) {
                uint32_t copyLen = thisLen * sizeof(tType);
                if (copyLen % 32 != 0) {
                    DataCopyExtParams dataCopyParamTail {1, copyLen, 0, 0, 0};
                    DataCopyPad(thisBatchSoftmaxGradGb[offset], copyOutLocalTensor, dataCopyParamTail);
                } else {
                    DataCopy(thisBatchSoftmaxGradGb[offset], copyOutLocalTensor, thisLen);
                }
            } else {
                DataCopyPad(thisBatchSoftmaxGradGb[offset], copyOutLocalTensor,  copyParams);
            }
            
            
            vecOutQueue.FreeTensor(copyOutLocalTensor);
            remain = remain - thisLen;
        }

    }
    
    private:
        NormGradArgs mmArgs;
        NormGradPipeArgs pipeArg;
        // C = A*B dA=C*BT (vec) dB=AT*C (Cube)
        TBuf<TPosition::VECCALC> tmpBuff;
        TQue<QuePosition::VECIN, 1> vecInQueue;
        TQue<QuePosition::VECIN, 1> vecInGradQueue;
        TQue<QuePosition::VECOUT, 1> vecOutQueue;
        
        GlobalTensor<tType> softmaxOut;
        GlobalTensor<tType> gradSoftmax;
        GlobalTensor<tType> gradValue;

};
#endif