#include "kernel_operator.h"
#include "all2all_uss.h"

extern "C" __global__ __aicore__ void lccl_all_uss(GM_ADDR send_data, GM_ADDR send_count_matrix, GM_ADDR shape_vec, GM_ADDR peer_mem, GM_ADDR restore, GM_ADDR rev_data, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    AllUss<float> opKernel(tiling_data.rank, tiling_data.rankSize, (1 << 2));
    opKernel.Init(send_data, send_count_matrix, shape_vec, peer_mem, restore, rev_data, tiling_data.rank, tiling_data.rankSize, tiling_data.magic, tiling_data.dim, tiling_data.outShape, tiling_data.ipc);

    opKernel.Process();
}