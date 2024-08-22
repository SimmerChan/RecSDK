/**
 * @file IndexSelectForRank1BackwardNpuApi.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
 
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
