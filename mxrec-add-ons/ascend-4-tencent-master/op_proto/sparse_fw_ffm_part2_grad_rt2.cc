/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "sparse_fw_ffm_part2_grad.h"
#include "runtime_util.h"
#include "op_util.h"

using namespace ge;
namespace ops {
constexpr size_t FW_IN_IDX = 3;
constexpr size_t CROSS_IN_IDX = 1;

ge::graphStatus InferShape4SparseFwFFMPart2Grad(gert::InferShapeContext* context) {
  auto fw_shape = context->GetInputShape(FW_IN_IDX);
  auto crosee_shape = context->GetInputShape(CROSS_IN_IDX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, fw_shape);
  OPS_CHECK_NULL_WITH_CONTEXT(context, crosee_shape);
  auto fw_res_shape = context->GetOutputShape(0);
  auto fw_cross_mean_sum_grad_shape = context->GetOutputShape(1);
  auto fw_cross_mean_square_sum_grad_shape = context->GetOutputShape(2);
  OPS_CHECK_NULL_WITH_CONTEXT(context, fw_res_shape);
  OPS_CHECK_NULL_WITH_CONTEXT(context, fw_cross_mean_sum_grad_shape);
  OPS_CHECK_NULL_WITH_CONTEXT(context, fw_cross_mean_square_sum_grad_shape);
  *fw_res_shape = *fw_shape;
  *fw_cross_mean_sum_grad_shape = *crosee_shape;
  *fw_cross_mean_square_sum_grad_shape = *crosee_shape;
  return GRAPH_SUCCESS;
}

IMPL_OP(SparseFwFFMPart2Grad).InferShape(InferShape4SparseFwFFMPart2Grad);
}  // namespace ops
