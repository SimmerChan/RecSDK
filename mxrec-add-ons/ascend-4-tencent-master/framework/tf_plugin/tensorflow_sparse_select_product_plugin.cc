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
 * \file sparse_select_product_plugin.cc
 * \brief
 */
#include <iostream>
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
// #include "proto/onnx/ge_onnx.pb.h"

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

Status ParseParamsSparseSelectProduct(const Message *op_src, ge::Operator& op_dest) {
    std::cout << "in ParseParamsSparseSelectProduct\n" << std::endl;
    // OP_LOGD(op_dest.GetName().c_str(), "in ParseParamsSparseSelectProduct");
    const domi::tensorflow::NodeDef *const node_src = ge::PtrToPtr<const ascend_private::protobuf::Message,
                                                      const domi::tensorflow::NodeDef>(op_src);
    //int n = node_src->input_size();
    std::cout << op_src << std::endl<< std::flush;
    std::cout << "in 59 " << std::endl<<std::flush;
    auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
    opDesc->AddDynamicInputDesc("x", 2);
    opDesc->AddDynamicOutputDesc("y", 1);

    // op_dest.SetAttr("original_type", "UnsortedSegmentSum");
    if (AutoMappingFn(op_src, op_dest) != SUCCESS) {
        return FAILED;
    }
    op_dest.SetAttr("original_type", "SparseSelectProduct");
    op_dest.SetAttr("name", node_src->name());
    std::cout << "leave ParseParamsSparseSelectProduct\n" << std::endl<<std::flush;
    // OP_LOGD(op_dest.GetName().c_str(), "leave ParseParamsSparseSelectProduct");
    return SUCCESS;
}

static Status ParseOpToGraphSparseSelectProduct(const ge::Operator &op, ge::Graph &graph) {
    // OP_LOGD(op.GetName().c_str(), "in ParseOpToGraphSparseSelectProduct");
    std::cout << "in ParseOpToGraphSparseSelectProduct\n" << std::endl<<std::flush;
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        // OP_LOGE(op.GetName().c_str(), "get name from op failed");
        return FAILED;
    }
    std::cout << "after 80" << std::endl<<std::flush;
    ge::Operator data = ge::op::Data(ori_name + "_weight").set_attr_index(0);
    // ge::Operator segment_ids = ge::op::Data(ori_name + "_input_egmentsegment_ids").set_attr_index(1);
    // ge::Operator num_segments = ge::op::Data(ori_name + "_input_num_segments").set_attr_index(2);
    ge::Operator for_mul = ge::op::Data(ori_name + "_index").set_attr_index(1);
    std::cout << "after 83" << std::endl<<std::flush;
    /*std::vector<int64_t> expand_axis = {0};
    *std::vector<int64_t> expand_axis_len = {1};
    *ge::Tensor expand_axis_tensor = Vec2Tensor(expand_axis, expand_axis_len, ge::DT_INT64);
    *ge::Operator expand_axis_op = ge::op::Const("expand_axis").set_attr_value(expand_axis_tensor);
    *ge::Operator expand_dims = ge::op::ExpandDims(ori_name + "_expand_dims")
    *                           .set_input_x(num_segments)
    *                           .set_input_axis(expand_axis_op);
    
    *int32_t n_num = 2;
    *std::vector<int64_t> axis_temp = {0};
    *std::vector<int64_t> axis_temp_len = {1};
    *ge::Tensor axis_temp_tensor = Vec2Tensor(axis_temp, axis_temp_len, ge::DT_INT64);
    *ge::Operator axis_temp_op = ge::op::Const("axis_temp").set_attr_value(axis_temp_tensor);
    *ge::Operator concat = ge::op::Concat(ori_name + "_concat")
    *                      .set_input_concat_dim(axis_temp_op)
    *                      .create_dynamic_input_x(n_num)
    *                      .set_dynamic_input_x(0, segment_ids)
    *                      .set_dynamic_input_x(1, expand_dims)
    *                      .set_attr_N(2);
    */
    
    std::vector<int32_t> mul_num = {1};
    std::vector<int64_t> mul_num_len = {1};
    ge::Tensor mul_num_tensor = Vec2Tensor(mul_num, mul_num_len, ge::DT_INT32);
    ge::Operator mul_num_op = ge::op::Const("mul_num").set_attr_value(mul_num_tensor);       
    ge::Operator concat = ge::op::Mul(ori_name + "_mul")
                          .set_input_x1(for_mul)
                          .set_input_x2(mul_num_op);

    ge::Operator shape_of_concat_result = ge::op::Shape(ori_name + "_shape")
                                          .set_input_x(concat);
    
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
                           .set_input_x(concat)
                           .set_input_offsets(offset_op)
                           .set_input_size(sub_len);
    std::cout << "after 123" << std::endl<<std::flush;
    std::vector<int32_t> size_num_2 = {1};
    std::vector<int64_t> size_num_2_len = {1};
    ge::Tensor size_num_2_tensor = Vec2Tensor(size_num_2, size_num_2_len, ge::DT_INT32);
    ge::Operator size_num_2_op = ge::op::Const("size_num_2_len").set_attr_value(size_num_2_tensor);
    ge::Operator slice_2 = ge::op::Slice(ori_name + "_slice_2")
                           .set_input_x(concat)
                           .set_input_offsets(sub_len)
                           .set_input_size(size_num_2_op);

    ge::Operator unsortedsegmentsum_raw = ge::op::UnsortedSegmentSum(ori_name + "_unsorted_segment_sum")
                                      .set_input_x(data)
                                      .set_input_segment_ids(slice)
                                      .set_input_num_segments(slice_2);
    std::cout << "after 137" << std::endl<<std::flush;
    std::vector<int64_t> axis_temp = {1};
    ge::Operator unsortedsegmentsum = ge::op::Squeeze(ori_name + "_squeeze_data")
                        .set_input_x(unsortedsegmentsum_raw)
                        .set_attr_axis(axis_temp);
    
    //std::vector<ge::Operator> inputs{data, segment_ids, num_segments};
    std::vector<ge::Operator> inputs{data, concat};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
    std::cout << "after 141" << std::endl<<std::flush;
    output_indexs.emplace_back(unsortedsegmentsum, vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(output_indexs);
    
    std::cout << "leave ParseOpToGraphSparseSelectProduct\n" << std::endl <<std::flush;
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseSelectProduct")
    // .OriginOpType("UnsortedSegmentSum")
    .ParseParamsFn(ParseParamsSparseSelectProduct)
    .ParseOpToGraphFn(ParseOpToGraphSparseSelectProduct)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
