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
 * \file sparse_select_product_grad_plugin.cc
 * \brief
 */
#include <string>
#include <vector>
#include <map>
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "proto/tensorflow/node_def.pb.h"
#include "graph/operator.h"
#include "graph/graph.h"
// #include "op_log.h"
#include "array_ops.h"
#include "split_combination_ops.h"
#include "selection_ops.h"
#include "elewise_calculation_ops.h"

using std::vector;

namespace domi {
// using namespace ge;

namespace {
template<typename T>
ge::Tensor Vec2Tensor(vector<T>& vals, const vector<int64_t>& dims, ge::DataType dtype, ge::Format format = ge::FORMAT_ND) {
  ge::Shape shape(dims);
  ge::TensorDesc desc(shape, format, dtype);
  ge::Tensor tensor(desc, reinterpret_cast<uint8_t*>(vals.data()), vals.size() * sizeof(T));
  return tensor;
}
}

Status ParseParamsSparseSelectProductGrad(const Message *op_src, ge::Operator& op_dest) {
    // OP_LOGD(op_dest.GetName().c_str(), "in ParseParamsSparseSelectProduct");
    const domi::tensorflow::NodeDef *const node_src = ge::PtrToPtr<const ascend_private::protobuf::Message,
                                                      const domi::tensorflow::NodeDef>(op_src);
    int n = node_src->input_size();
    auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
    opDesc->AddDynamicInputDesc("x", 2);
    opDesc->AddDynamicOutputDesc("y", 1);
    if (AutoMappingFn(op_src, op_dest) != SUCCESS) {
        return FAILED;
    }  
    // OP_LOGD(op_dest.GetName().c_str(), "leave ParseParamsSparseSelectProduct");

    // op_dest.SetAttr("original_type", "GatherV2");
    op_dest.SetAttr("original_type", "SparseSelectProductGrad");
    op_dest.SetAttr("name", node_src->name());
    return SUCCESS;
}

static Status ParseOpToGraphSparseSelectProductGrad(const ge::Operator &op, ge::Graph &graph) {
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        // OP_LOGE(op.GetName().c_str(), "get name from op failed");
        return FAILED;
    }

    ge::Operator data_0 = ge::op::Data(ori_name + "_input_data_0").set_attr_index(0);
    ge::Operator data_1 = ge::op::Data(ori_name + "_input_data_1").set_attr_index(1);
    // ge::Operator data_2 = ge::op::Data(ori_name + "_input_data_2").set_attr_index(2);

    std::vector<int32_t> mul_num = {1};
    std::vector<int64_t> mul_num_len = {1};
    ge::Tensor mul_num_tensor = Vec2Tensor(mul_num, mul_num_len, ge::DT_INT32);
    ge::Operator mul_num_op = ge::op::Const("mul_num").set_attr_value(mul_num_tensor);       
    ge::Operator mul_re = ge::op::Mul(ori_name + "_mul")
                          .set_input_x1(data_0)
                          .set_input_x2(mul_num_op);

    ge::Operator shape_of_concat_result = ge::op::Shape(ori_name + "_shape")
                                          .set_input_x(mul_re);        

    std::vector<int32_t> subtrahend = {1};
    std::vector<int64_t> subtrahend_len = {1};
    ge::Tensor subtrahend_tensor = Vec2Tensor(subtrahend, subtrahend_len, ge::DT_INT32);
    ge::Operator subtrahend_op = ge::op::Const("subtrahend").set_attr_value(subtrahend_tensor);
    ge::Operator sub_len = ge::op::Sub(ori_name + "_sub")
                           .set_input_x1(shape_of_concat_result)
                           .set_input_x2(subtrahend_op);

    std::vector<int32_t> offset = {0};
    std::vector<int64_t> offset_len = {1};
    ge::Tensor offset_tensor = Vec2Tensor(offset, offset_len, ge::DT_INT32);
    ge::Operator offset_op = ge::op::Const("offset").set_attr_value(offset_tensor);
    ge::Operator slice = ge::op::Slice(ori_name + "_slice")
                           .set_input_x(mul_re)
                           .set_input_offsets(offset_op)
                           .set_input_size(sub_len);

    ge::Operator gather = ge::op::Gather(ori_name + "_gather")
                          .set_input_x(data_1)
                          .set_input_indices(slice);

    std::vector<int64_t> expand_axis = {1};
    std::vector<int64_t> expand_axis_len = {1};
    ge::Tensor expand_axis_tensor = Vec2Tensor(expand_axis, expand_axis_len, ge::DT_INT64);
    ge::Operator expand_axis_op = ge::op::Const("expand_axis").set_attr_value(expand_axis_tensor);
    ge::Operator expand_dims = ge::op::ExpandDims(ori_name + "_expand_dims")
                               .set_input_x(gather)
                               .set_input_axis(expand_axis_op);
    
    // std::vector<ge::Operator> inputs{data_0, data_1, data_2};
    std::vector<ge::Operator> inputs{data_0, data_1};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
    output_indexs.emplace_back(expand_dims, vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseSelectProductGrad")
    // .OriginOpType("GatherV2")
    .ParseParamsFn(ParseParamsSparseSelectProductGrad)
    .ParseOpToGraphFn(ParseOpToGraphSparseSelectProductGrad)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
