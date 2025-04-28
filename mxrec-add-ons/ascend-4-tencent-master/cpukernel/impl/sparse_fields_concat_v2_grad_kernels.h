
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: api of SparseFieldsConcatV2Grad
 */

#ifndef _SPARSE_FIELDS_CONCAT_V2_GRAD_KERNELS_H_
#define _SPARSE_FIELDS_CONCAT_V2_GRAD_KERNELS_H_

#include "cpu_kernel.h"
#include <vector>



namespace aicpu {
class SparseFieldsConcatV2GradCpuKernel : public CpuKernel {
public:
    ~SparseFieldsConcatV2GradCpuKernel() = default;
    virtual uint32_t Compute(CpuKernelContext &ctx) override;
};
} // namespace aicpu
#endif
