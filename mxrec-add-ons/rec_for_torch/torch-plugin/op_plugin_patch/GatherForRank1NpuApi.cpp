/* Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.

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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor gather_for_rank1(const at::Tensor& x, const at::Tensor& index)
{
    TORCH_CHECK(x.dim() == 1, "The x should be 1D", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(index.dim() == 1, "The index should be 1D", OPS_ERROR(ErrCode::PARAM));
    auto xConti = x.contiguous();
    auto indexConti = index.contiguous();
    at::Tensor y = at::empty({indexConti.size(0)}, xConti.options());
    EXEC_NPU_CMD(aclnnGatherForRank1, xConti, indexConti, y);
    return y;
}
}  // namespace op_api
