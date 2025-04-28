
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: api of SparseFM
 */

#ifndef _SPARSE_FM_KERNELS_H_
#define _SPARSE_FM_KERNELS_H_

#include "cpu_kernel.h"

namespace aicpu {
class SparseFMCpuKernel : public CpuKernel {
public:
    ~SparseFMCpuKernel() = default;
    virtual uint32_t Compute(CpuKernelContext &ctx) override;
};

} // namespace aicpu
#endif
