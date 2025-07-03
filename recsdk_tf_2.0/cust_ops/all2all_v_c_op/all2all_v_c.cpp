
/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "all2all_v_c.h"

#include "kernel_operator.h"
#include "op_def.h"
#include "collectives.h"

#include "all2all_v_c_cf.h"

#define ALL2ALL_CLASS_OP_LAUNCH(name, type)             \
    do {                                                \
        name<type> opKernel(rank, rankSize, extraFlag); \
        opKernel.Init(KERNELS_ARGS_CALL());          \
        opKernel.Process();                             \
    } while (0)

#define LCCL_ALL2ALLVC_FUNC_AUTO_DEF(type)                                            \
    extern "C" __global__ __aicore__ void LcalAll2AllVC_##type(KERNELS_ARGS_FUN()) \
    {                                                                                 \
        GET_COMM_ARGS;                                                                \
        constexpr int32_t smallYRankSize = 8;                                         \
        if ((rankSize/localRankSize) <= smallYRankSize) {                             \
            ALL2ALL_CLASS_OP_LAUNCH(All2AllVC, type);                            \
        } else {                                                                      \
            ALL2ALL_CLASS_OP_LAUNCH(All2AllVCCF, type);                        \
        }                                                                             \
    }

#if defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__) || defined(__DAV_L311__)
LCCL_910_95_TYPE_FUNC(LCCL_ALL2ALLVC_FUNC_AUTO_DEF);
#endif
