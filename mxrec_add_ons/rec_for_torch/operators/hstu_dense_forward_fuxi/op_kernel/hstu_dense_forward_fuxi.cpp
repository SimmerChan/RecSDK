#include "hstu_dense_forward_normal_kernel_v200_fuxi.h"

#include "kernel_operator.h"

extern "C" __global__ __aicore__ void hstu_dense_forward_fuxi(GM_ADDR q, GM_ADDR k, GM_ADDR v,
    GM_ADDR timestampBias, GM_ADDR positionBias, GM_ADDR mask, GM_ADDR attnOutput, GM_ADDR workspace, GM_ADDR tiling)
{
    HstuDenseForwardFuxi::Args args{q, k, v, timestampBias, positionBias, mask, attnOutput, workspace, tiling};
    if (TILING_KEY_IS(0)) {
        INVOKE_HSTU_NORMAL_V200_FUXI_OP_IMPL(half);
    }
}