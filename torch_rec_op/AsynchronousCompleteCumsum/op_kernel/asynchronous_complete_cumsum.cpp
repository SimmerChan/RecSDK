#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void asynchronous_complete_cumsum(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                                                   GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    int64_t totalLen = tiling_data.totalLength;
    uint32_t inputType = tiling_data.inputType;

    switch (inputType)
    {
        case 0 :
        {
            __gm__ int64_t* xPtr = (__gm__ int64_t*) x;
            __gm__ int64_t* yPtr = (__gm__ int64_t*) y;
            *(yPtr) = 0;
            for (int i=0; i<totalLen+1; i++) {
                *(yPtr+i) = *(xPtr+i-1) + *(yPtr+i-1);
            }
            break;
        }

        case 1:
        {
            __gm__ int32_t* xPtr = (__gm__ int32_t*) x;
            __gm__ int32_t* yPtr = (__gm__ int32_t*) y;
            *(yPtr) = 0;
            for (int i=0; i<totalLen+1; i++) {
                *(yPtr+i) = *(xPtr+i-1) + *(yPtr+i-1);
            }
            break;
        }
    }
}