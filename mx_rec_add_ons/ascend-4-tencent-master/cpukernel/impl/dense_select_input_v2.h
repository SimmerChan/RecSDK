
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: api of SparseFM
 */

#ifndef _DENSE_SELECT_INPUT_V2_H_
#define _DENSE_SELECT_INPUT_V2_H_

#include "cpu_kernel.h"
#include <vector>

namespace aicpu {
class DenseSelectInputV2CpuKernel : public CpuKernel {
public:
    ~DenseSelectInputV2CpuKernel() = default;
    virtual uint32_t Compute(CpuKernelContext &ctx) override;
    uint32_t ImplCompute(CpuKernelContext &ctx);
};
} // namespace aicpu
#endif
