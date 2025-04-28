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

std::tuple<at::Tensor, at::Tensor> index_select_for_rank1_backward(const at::Tensor& gradY, const at::Tensor& x,
                                                                   const at::Tensor& index)
{
    auto dense_gradY = gradY.contiguous();
    auto dense_index = index.contiguous();
    at::Tensor gradX = at::zeros_like(x);
    at::Tensor gradIndex = at::zeros_like(index);
    EXEC_NPU_CMD(aclnnIndexSelectForRank1Backward, dense_gradY, x, dense_index, gradX, gradIndex);
    return std::make_tuple(gradX, gradIndex);
}
}  // namespace op_api
