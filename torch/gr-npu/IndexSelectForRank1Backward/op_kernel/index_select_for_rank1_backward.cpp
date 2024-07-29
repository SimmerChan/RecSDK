#include <cstdint>
#include "kernel_operator.h"
#include "utils.h"
#ifdef __CCE_KT_TEST__
#include "index_select_for_rank1_backward_tiling.h"
#endif

using namespace AscendC;
extern "C" __global__ __aicore__ void index_select_for_rank1_backward(GM_ADDR gradY, GM_ADDR x, GM_ADDR index, GM_ADDR gradX, GM_ADDR gradIndex, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    LOG("kernel run sucessful");

    int64_t totalLen = tiling_data.totalLen;
    int64_t xDim0 = tiling_data.xDim0;
    int64_t baseLen = tiling_data.baseLen;
    int64_t tailSplitIndex = tiling_data.tailSplitIndex;
    int64_t keyDim0Align64B = tiling_data.keyDim0Align64B;

    int64_t lenOfThisCore;
    int64_t offsetOfThisCore;
    if (GetBlockIdx() >= tailSplitIndex) {
        lenOfThisCore = baseLen;
        offsetOfThisCore = tailSplitIndex*(baseLen+1) + (GetBlockIdx()-tailSplitIndex)*baseLen;
    } else {
        lenOfThisCore = baseLen+1;
        offsetOfThisCore = GetBlockIdx()*(baseLen+1);
    }

    LOG("lenOfThisCore ", lenOfThisCore, " offsetOfThisCore", offsetOfThisCore);

    __gm__ int64_t* indexPtr = (__gm__ int64_t*) index;
    __gm__ float* gradYPtr = (__gm__ float*) gradY;
    __gm__ float* gradXPtr = (__gm__ float*) gradX;
    __gm__ float* resultPtr = (__gm__ float*) workspace + GetBlockIdx()*keyDim0Align64B;
    
    // 将workspace中的值清空
    for (int i = 0; i < keyDim0Align64B ; i++) {
        *(resultPtr+i) = 0.0;
    }

    for (int64_t i = offsetOfThisCore; i < offsetOfThisCore + lenOfThisCore; i++) {
        int64_t thisIndex = *(indexPtr+i);
        float thisGradX = *(gradYPtr + i);
        *(resultPtr+thisIndex) +=  thisGradX;
    }
    
    GlobalTensor<float> resultGm;
    resultGm.SetGlobalBuffer((__gm__ float *) workspace, keyDim0Align64B*GetBlockNum());

    // 注意此函数不支持910A, 所以此算子不支持910A
    SyncAll();
    DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(resultGm);

    if (GetBlockIdx()==0) {
        __gm__ float * allResultPtr = (__gm__ float *) workspace;
        for (int j = 0; j < xDim0 ; j++) {
            *(gradXPtr+j) = 0.0;
        }
        for (int i = 0; i < GetBlockNum(); i++) {
            for (int j = 0; j < xDim0 ; j++) {
                *(gradXPtr+j) += resultGm.GetValue(i*keyDim0Align64B+j); //*(allResultPtr+i*keyDim0Align64B+j);
            }
            
        }
    }
}