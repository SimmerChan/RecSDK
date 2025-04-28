
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: api of DenseSelectInput
 */

#ifndef _DENSE_SELECT_INPUT_KERNELS_H_
#define _DENSE_SELECT_INPUT_KERNELS_H_

#include "cpu_kernel.h"

namespace aicpu {
class DenseSelectInputCpuKernel : public CpuKernel {
public:
    ~DenseSelectInputCpuKernel() = default;
    virtual uint32_t Compute(CpuKernelContext &ctx) override;
};
} // namespace aicpu
#endif
