#include "gather_for_rank1_kernel.h"
#include "kernel_operator.h"
extern "C" __global__ __aicore__ void gather_for_rank1(GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR workspace,
                                                       GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    GatherForRank1::Args args{x, index, y, workspace, tiling};

    if (TILING_KEY_IS(0)) {
        GatherForRank1::GatherForRank1Kernel<float> kernel(args);
        kernel.Compute();
    } else if (TILING_KEY_IS(1)) {
        GatherForRank1::GatherForRank1Kernel<half> kernel(args);
        kernel.Compute();
    }
}