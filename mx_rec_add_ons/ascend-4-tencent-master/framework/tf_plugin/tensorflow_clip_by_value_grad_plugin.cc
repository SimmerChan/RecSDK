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

/*!
 * \file clip_by_value_grad_plugin.cc
 * \brief
 */
#include <string>
#include <vector>
#include <map>
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/operator.h"
#include "graph/graph.h"
// #include "op_log.h"
#include "array_ops.h"
#include "elewise_calculation_ops.h"

namespace domi {
using namespace ge;
Status ParseParamsAddsMuls(const ge::Operator& op_origin, ge::Operator& op_dest) {
    AutoMappingByOpFn(op_origin, op_dest);
    op_dest.SetAttr("original_type", "ClipByValueGrad");
    // op_dest.SetAttr("original_type", "Ceil");

    float value_1 = 1.0;
    float value_2 = 0.0;

    std::shared_ptr<ge::OpDesc> op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
    op_desc->AddDynamicInputDesc("args", 1);
    op_desc->AddDynamicOutputDesc("output", 1);
    op_dest.SetAttr("value_1", value_1);
    op_dest.SetAttr("value_2", value_2);
    return SUCCESS;
}

static Status ParseOpToGraphAddsMuls(const ge::Operator &op, ge::Graph &graph) {
    auto data_0 = ge::op::Data().set_attr_index(0);
    float value_1 = 1.0;
    float value_2 = 0.0;
    if (op.GetAttr("value_1", value_1) != ge::GRAPH_SUCCESS) {
        // OP_LOGE(TbeGetName(op).c_str(), "get attr value_1 failed.");
        return FAILED;
    }
    if (op.GetAttr("value_2", value_2) != ge::GRAPH_SUCCESS) {
        // OP_LOGE(TbeGetName(op).c_str(), "get attr value_2 failed.");
        return FAILED;
    }

    auto muls = ge::op::Muls("Muls");
    muls.set_input_x(data_0);
    muls.set_attr_value(value_1);
    auto adds = ge::op::Adds("Adds");
    adds.set_input_x(muls);
    adds.set_attr_value(value_2);

    std::vector<ge::Operator> inputs{data_0};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
    output_indexs.emplace_back(adds, vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ClipByValueGrad")
    // .OriginOpType("Ceil")
    .ParseParamsByOperatorFn(ParseParamsAddsMuls)
    .ParseOpToGraphFn(ParseOpToGraphAddsMuls)
    .ImplyType(ImplyType::TVM);
}  // namespace domi