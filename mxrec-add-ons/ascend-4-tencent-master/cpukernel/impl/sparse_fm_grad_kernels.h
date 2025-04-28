
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: api of SparseFMGrad
 */

#ifndef _SPARSE_FM_GRAD_KERNELS_H_
#define _SPARSE_FM_GRAD_KERNELS_H_

#include "cpu_kernel.h"

namespace aicpu {
class SparseFMGradCpuKernel : public CpuKernel {
public:
    ~SparseFMGradCpuKernel() = default;
    virtual uint32_t Compute(CpuKernelContext &ctx) override;
};
} // namespace aicpu
#endif
