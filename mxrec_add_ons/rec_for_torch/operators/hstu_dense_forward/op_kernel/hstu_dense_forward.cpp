#ifdef SUPPORT_V200
#include "hstu_dense_forward_normal_kernel_v200.h"

template <typename T>
__aicore__ void InvokeHstuOpImpl(const HstuDenseForward::Args &args)
{
    TPipe tPipe;
    T op;
    GET_TILING_DATA(tilingData, args.tiling);
    const HstuDenseForwardTilingData *__restrict tilingDataPtr = &tilingData;
    REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.qkMatmul, &tilingDataPtr->qkMatmul, op.svMatmul,
                      &tilingDataPtr->svMatmul);
    op.Init(args, tilingDataPtr, &tPipe);
    op.Compute(tilingDataPtr);
}

#else
#include "hstu_dense_forward_jagged_kernel.h"
#include "hstu_dense_forward_normal_kernel.h"

template <typename T>
__aicore__ void InvokeHstuOpImpl(const HstuDenseForward::Args &args)
{
    TPipe tPipe;
    T op;
    GET_TILING_DATA(tilingData, args.tiling);
    const HstuDenseForwardTilingData *__restrict tilingDataPtr = &tilingData;
    REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.qkMatmul, &tilingDataPtr->qkMatmul, op.svMatmul,
                      &tilingDataPtr->svMatmul);
    uint64_t tilingPtr = reinterpret_cast<uint64_t>(args.tiling);
    op.qkMatmul.SetUserDefInfo(tilingPtr);
    op.svMatmul.SetUserDefInfo(tilingPtr);
    op.Init(args, tilingDataPtr, &tPipe);
    op.Compute(tilingDataPtr);
}

#endif

#include "kernel_operator.h"

extern "C" __global__ __aicore__ void hstu_dense_forward(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask,
                                                         GM_ADDR attnBias, GM_ADDR attnOutput, GM_ADDR workspace,
                                                         GM_ADDR tiling)
{
    HstuDenseForward::Args args{q, k, v, attnBias, mask, attnOutput, workspace, tiling};
#ifdef SUPPORT_V200
    if (TILING_KEY_IS(0)) {
        InvokeHstuOpImpl<HstuDenseForward::HstuDenseForwardKernelv200<half>>(args);
    }
#else
    if (TILING_KEY_IS(0)) {
        InvokeHstuOpImpl<HstuDenseForward::HstuDenseForwardKernel<half>>(args);
    } else if (TILING_KEY_IS(1)) {
        InvokeHstuOpImpl<HstuDenseForward::HstuDenseForwardKernel<bfloat16_t>>(args);
    } else if (TILING_KEY_IS(2)) {
        InvokeHstuOpImpl<HstuDenseForward::HstuDenseForwardKernel<float>>(args);
    } else if (TILING_KEY_IS(3)) {
        InvokeHstuOpImpl<HstuDenseForward::HstuDenseForwardJaggedKernel<half>>(args);
    } else if (TILING_KEY_IS(4)) {
        InvokeHstuOpImpl<HstuDenseForward::HstuDenseForwardJaggedKernel<bfloat16_t>>(args);
    } else if (TILING_KEY_IS(5)) {
        InvokeHstuOpImpl<HstuDenseForward::HstuDenseForwardJaggedKernel<float>>(args);
    }
#endif
}