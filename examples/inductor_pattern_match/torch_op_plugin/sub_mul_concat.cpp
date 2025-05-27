/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/

#include "torch_npu_helper.h"

namespace acl_ops {
static constexpr int NUM_CONCAT = 4;
static constexpr int DIM_LAST = 2;

/// This operator will concat four tensors: lhs, rhs, lhs - rhs, lhs * rhs, along the last dim.
/// Constraints:
/// 1. dims of lhs and rhs must be 3.
at::Tensor SubMulConcat(const at::Tensor& lhs, const at::Tensor& rhs)
{
    const at::OptionalDeviceGuard guard(device_of(lhs));
    auto output = at::empty({lhs.size(0), lhs.size(1), NUM_CONCAT * lhs.size(DIM_LAST)}, lhs.options());
    EXEC_NPU_CMD(aclnnSubMulConcat, lhs, rhs, output);
}
}  // namespace acl_ops

TORCH_LIBRARY_FRAGMENT(acl_ops, m)
{
    m.def("sub_mul_concat(Tensor lhs, Tensor rhs) -> Tensor");
}

TORCH_LIBRARY_IMPL(acl_ops, PrivateUse1, m)
{
    m.impl("sub_mul_concat", &acl_ops::SubMulConcat);
}