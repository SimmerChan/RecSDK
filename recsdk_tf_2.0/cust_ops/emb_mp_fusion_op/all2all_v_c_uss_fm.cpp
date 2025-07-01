/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#include "kernel_operator.h"
#include "op_def.h"
#include "collectives.h"

#include "all2all_v_c_uss_fm.h"

#define ALL2ALL_USS_FM_CLASS_OP_LAUNCH(name, type)         \
    do {                                                \
        name<type> opKernel(rank, rankSize, extraFlag); \
        opKernel.Init(EMB_FUSION_ARGS_CALL());          \
        opKernel.Process();                             \
    } while (0)

#define LCCL_ALL2ALLVC_USS_FM_FUNC_AUTO_DEF(type)                                            \
    extern "C" __global__ __aicore__ void LcalAll2AllVC_USS_FM_##type(EMB_FUSION_ARGS_FUN()) \
    {                                                                                     \
        GET_COMM_ARGS;                                                                    \
        ALL2ALL_USS_FM_CLASS_OP_LAUNCH(All2AllVCUSSFM, type);                             \
    }

LCCL_ALL2ALLVC_USS_FM_FUNC_AUTO_DEF(float)