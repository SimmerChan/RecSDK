#ifdef SUPPORT_V200
    #include "hstu_dense_forward_normal_kernel_v200.h"
#else
    #include "hstu_dense_forward_jagged_kernel.h"
    #include "hstu_dense_forward_normal_kernel.h"
#endif

#include "kernel_operator.h"

extern "C" __global__ __aicore__ void hstu_dense_forward(GM_ADDR q, GM_ADDR k, GM_ADDR v,
                                                         GM_ADDR mask, GM_ADDR attnBias,
                                                         GM_ADDR attnOutput, GM_ADDR workspace, GM_ADDR tiling)
{
    HstuDenseForward::Args args{q, k, v, attnBias, mask, attnOutput, workspace, tiling};
#ifdef SUPPORT_V200
    if (TILING_KEY_IS(0)) {
        INVOKE_HSTU_NORMAL_V200_OP_IMPL(half);
    }
#else
    if (TILING_KEY_IS(0)) {
        INVOKE_HSTU_NORMAL_OP_IMPL(half);
    } else if (TILING_KEY_IS(1)) {
        INVOKE_HSTU_NORMAL_OP_IMPL(bfloat16_t);
    } else if (TILING_KEY_IS(2)) {
        INVOKE_HSTU_NORMAL_OP_IMPL(float);
    } else if (TILING_KEY_IS(3)) {
        INVOKE_HSTU_JAGGED_OP_IMPL(half);
    } else if (TILING_KEY_IS(4)) {
        INVOKE_HSTU_JAGGED_OP_IMPL(bfloat16_t);
    } else if (TILING_KEY_IS(5)) {
        INVOKE_HSTU_JAGGED_OP_IMPL(float);
    }
#endif
}