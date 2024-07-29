#include <cstdint>
#include "kernel_operator.h"
#include "utils.h"
#ifdef __CCE_KT_TEST__
#include "jagged_to_padded_dense_tiling.h"
#endif

constexpr int DATA_TYPE_INT64=8;
using namespace AscendC;
extern "C" __global__ __aicore__ void jagged_to_padded_dense(GM_ADDR values, GM_ADDR offsets, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    int64_t totalBatch = tiling_data.totalBatch;
    int64_t baseBatchLen = tiling_data.baseBatchLen;
    int64_t tailSplitIndex = tiling_data.tailSplitIndex;
    int64_t valuesDim0 = tiling_data.valuesDim0;
    int64_t valuesDim1 = tiling_data.valuesDim1;
    int64_t offsetDim0 = tiling_data.offsetDim0;
    int64_t outDim1 = tiling_data.outDim1;
    int64_t ubCanUsed = tiling_data.ubCanUsed;
    int64_t bytesOfDataType = tiling_data.bytesOfDataType;
    int64_t offsetDataType = tiling_data.offsetDataType;

    // 计算出此核的偏移
    int64_t lenOfThisCore;
    int64_t offsetOfThisCore;
    if (GetBlockIdx() >= tailSplitIndex) {
        lenOfThisCore = baseBatchLen;
        offsetOfThisCore = tailSplitIndex*(baseBatchLen+1) + (GetBlockIdx()-tailSplitIndex)*baseBatchLen;
    } else {
        lenOfThisCore = baseBatchLen+1;
        offsetOfThisCore = GetBlockIdx()*(baseBatchLen+1);
    }

    GlobalTensor<uint8_t> valuesGT;
    GlobalTensor<uint8_t> outGT;
    valuesGT.SetGlobalBuffer(values, valuesDim0*valuesDim1*bytesOfDataType);
    outGT.SetGlobalBuffer(out, offsetDim0*outDim1*valuesDim1*bytesOfDataType);

    // 初始化pipe
    TPipe pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> inQueueX;
    pipe.InitBuffer(inQueueX, 2, ubCanUsed/2);
    int64_t blockLen = ubCanUsed/2;

    // 遍历所有的offset
    for(int64_t i = offsetOfThisCore ; i < lenOfThisCore+offsetOfThisCore; i++) {
        int64_t offsetThisIndex;
        int64_t offsetNextIndex;
        if (offsetDataType == DATA_TYPE_INT64) {
                // 初始化所有向量
            __gm__ int64_t* offsetsPtr = (__gm__ int64_t*) offsets;
            offsetThisIndex = *(offsetsPtr+i);
            offsetNextIndex = *(offsetsPtr+i+1);
        } else {
            __gm__ int32_t* offsetsPtr = (__gm__ int32_t*) offsets;
            offsetThisIndex = *(offsetsPtr+i);
            offsetNextIndex = *(offsetsPtr+i+1);     
        }
        int64_t valuesStartIndex = offsetThisIndex*valuesDim1*bytesOfDataType;
        int64_t valuesEndIndex = offsetNextIndex*valuesDim1*bytesOfDataType;

        int64_t outStartIndex = i*valuesDim1*outDim1*bytesOfDataType;
        int64_t outEndIndex = (i+1)*valuesDim1*outDim1*bytesOfDataType;

        if ((valuesEndIndex-valuesStartIndex)<0) {
            continue;
        }

        if ((valuesEndIndex-valuesStartIndex)>(outEndIndex-outStartIndex)) {
            valuesEndIndex = valuesStartIndex + outEndIndex-outStartIndex;
        }

        int64_t totalLen = valuesEndIndex-valuesStartIndex;
        int64_t remainLen = totalLen;
        while(remainLen > 0) {
            int64_t thisLen = blockLen;
            if (remainLen < blockLen) {
                thisLen = remainLen;
            }
            LocalTensor<uint8_t> localTensor = inQueueX.AllocTensor<uint8_t>();

            // 首先用datacopy拷贝大部分数据，最后用DataCopyPad拷贝末尾的数据
            uint32_t alignLen = thisLen/32*32; 
            uint32_t unAlignLen = thisLen-thisLen/32*32;

            // 对齐拷贝
            DataCopy(localTensor, valuesGT[valuesStartIndex], alignLen);
            if (unAlignLen!=0) {
                const DataCopyExtParams  dataCopyExtParams {1, unAlignLen, 0, 0, 0};

                const DataCopyPadExtParams<uint8_t> dataCopyPadExtParams {false, 0, 0, 0};
                DataCopyPad(localTensor[alignLen], valuesGT[valuesStartIndex+alignLen], dataCopyExtParams, dataCopyPadExtParams);
            }
            inQueueX.EnQue(localTensor);

            LocalTensor<uint8_t> outPutTensor = inQueueX.DeQue<uint8_t>();

            // 对齐拷贝
            DataCopy(outGT[outStartIndex], outPutTensor, alignLen);
            if (unAlignLen!=0) {
                const DataCopyExtParams  dataCopyExtParams {1, unAlignLen, 0, 0, 0};
                DataCopyPad(outGT[outStartIndex+alignLen], outPutTensor[alignLen], dataCopyExtParams);
            }

            outStartIndex += thisLen;
            valuesStartIndex += thisLen;
            inQueueX.FreeTensor(outPutTensor);
            remainLen = remainLen - thisLen;
        }
    }
}