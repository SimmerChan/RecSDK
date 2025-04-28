#include <cstdint>

#include "kernel_operator.h"

#include "hstu_dense_backward_jagged_kernel.h"
#include "hstu_dense_backward_kernel.h"

extern "C" __global__ __aicore__ void hstu_dense_backward(
    GM_ADDR grad, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR attnBias,
    GM_ADDR qGrad, GM_ADDR kGrad, GM_ADDR vGrad, GM_ADDR attnBiasGrad,
    GM_ADDR workspace, GM_ADDR tiling)
{
    HstuDenseBackward::Args args{grad, q, k, v, mask, attnBias, qGrad, kGrad, vGrad, attnBiasGrad, workspace, tiling};

    if (TILING_KEY_IS(5)) {
        HstuDenseBackward::HstuDenseBackwardJaggedKernel<float> kernel;
        kernel.Compute(args);
    } else if (TILING_KEY_IS(4)) {
        HstuDenseBackward::HstuDenseBackwardJaggedKernel<bfloat16_t> kernel;
        kernel.Compute(args);
    } else if (TILING_KEY_IS(3)) {
        HstuDenseBackward::HstuDenseBackwardJaggedKernel<half> kernel;
        kernel.Compute(args);
    } else if (TILING_KEY_IS(2)) {
        HstuDenseBackward::HstuDenseBackwardKernel<float> kernel;
        kernel.Compute(args);
    } else if (TILING_KEY_IS(1)) {
        HstuDenseBackward::HstuDenseBackwardKernel<bfloat16_t> kernel;
        kernel.Compute(args);
    } else if (TILING_KEY_IS(0)) {
        HstuDenseBackward::HstuDenseBackwardKernel<half> kernel;
        kernel.Compute(args);
    }
}