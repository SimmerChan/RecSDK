
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: api of SparseFwFFMGrad
 */

#ifndef _SPARSE_FW_FFM_GRAD_KERNELS_H_
#define _SPARSE_FW_FFM_GRAD_KERNELS_H_

#include "cpu_kernel.h"

namespace aicpu {
class SparseFwFFMGradCpuKernel : public CpuKernel {
public:
    ~SparseFwFFMGradCpuKernel() = default;
    virtual uint32_t Compute(CpuKernelContext &ctx) override;
};
} // namespace aicpu
#endif
