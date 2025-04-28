
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: api of SparseFieldsConcatV2
 */

#ifndef _SPARSE_FIELDS_CONCAT_V2_KERNELS_H_
#define _SPARSE_FIELDS_CONCAT_V2_KERNELS_H_

#include "cpu_kernel.h"

namespace aicpu {
class SparseFieldsConcatV2CpuKernel : public CpuKernel {
public:
    ~SparseFieldsConcatV2CpuKernel() = default;
    virtual uint32_t Compute(CpuKernelContext &ctx) override;
};
} // namespace aicpu
#endif