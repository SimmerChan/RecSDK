#include "kernel_operator.h"
#include "all2all.h"

extern "C" __global__ __aicore__ void lccl_all_to_all(GM_ADDR send_data, GM_ADDR send_count_matrix, GM_ADDR shape_vec, GM_ADDR peer_mem, GM_ADDR rev_data, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    All2All<float> opKernel(tiling_data.rank, tiling_data.rankSize, (1 << 2));
    opKernel.Init(send_data, send_count_matrix, shape_vec, peer_mem, rev_data, tiling_data.rank, tiling_data.rankSize, tiling_data.magic);

    opKernel.Process();
}