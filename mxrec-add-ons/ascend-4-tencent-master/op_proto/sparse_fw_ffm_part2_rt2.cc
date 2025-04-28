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
#include "sparse_fw_ffm_part2.h"
#include "runtime_util.h"
#include "op_util.h"
#include "op_log.h"
#include "op_const.h"

using namespace ge;
namespace ops {
// -------------------SparseFwFFMPart2 Ops START---------------------
ge::graphStatus Infershape4SparseFwFFMPart2(gert::InferShapeContext* context) {
  OP_LOGD(context->GetNodeName(), "Begin to do SparseFwFFMPart2Infershape.");
  const gert::Shape* cross_mean_sum_shape = context->GetInputShape(1);
  OPS_CHECK_NULL_WITH_CONTEXT(context, cross_mean_sum_shape);

  
  gert::Shape* output_shape = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, output_shape);
  int64_t batch_size = cross_mean_sum_shape->GetDim(0);
  int64_t embedding_size = cross_mean_sum_shape->GetDim(3);

  output_shape->SetDimNum(0);
  output_shape->AppendDim(batch_size);
  output_shape->AppendDim(embedding_size);

  OP_LOGD(context->GetNodeName(), "output_shape = %s.", ToString(*output_shape).c_str());
  OP_LOGD(context->GetNodeName(), "End to do SparseFwFFMPart2Infershape.");

  return ge::GRAPH_SUCCESS;
}

IMPL_OP(SparseFwFFMPart2).InferShape(Infershape4SparseFwFFMPart2);
}
// -------------------SparseFwFFMPart2 Ops END---------------------
