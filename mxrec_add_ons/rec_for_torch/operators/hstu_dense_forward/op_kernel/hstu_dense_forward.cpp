/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/


#ifdef SUPPORT_V200
    #include "hstu_dense_forward_kernel_v200.h"
#else
    #include "hstu_dense_forward_jagged_kernel.h"
    #include "hstu_dense_forward_kernel.h"
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