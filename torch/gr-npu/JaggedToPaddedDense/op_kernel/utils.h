/**
 * @file utils.h
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ATTENTION_FUSION_GRAD_UTILS_H
#define ATTENTION_FUSION_GRAD_UTILS_H
#include "kernel_operator.h"

template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 a, T2 b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

#ifdef __CCE_KT_TEST__

#define LOG(X...) Log(X)
__global__ __aicore__ void printArgs() {}

template <typename T, typename... Args>
__global__ __aicore__ void printArgs(T t, Args&&... args)
{
    std::cout << t << " ";
    printArgs(args...);
}

template <typename... Args>
__global__ __aicore__ void Log(Args&&... args)
{
    // don't log when using npu
    std::cout << "[AttentionFusion LOG][" << AscendC::GetBlockIdx() << "]  ";
    printArgs(args...);
    std::cout << std::endl;
}

#else
#define LOG(X...)
#endif
#endif