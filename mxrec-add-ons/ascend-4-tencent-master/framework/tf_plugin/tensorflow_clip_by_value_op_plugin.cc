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
 * \file clip_by_value_op_plugin.cc
 * \brief
 */
#include <string>
#include <vector>
#include <map>
#include "register/register.h"
#include "graph/operator.h"
#include "graph/graph.h"
// #include "op_log.h"
#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "proto/tensorflow/node_def.pb.h"
#include "graph/utils/op_desc_utils.h"
#include "selection_ops.h"
// #include "unsupported/Eigen/CXX11/Tensor"

namespace domi {
using namespace ge;
template<typename T>
ge::Tensor Vec2Tensor(vector<T>& vals, const vector<int64_t>& dims, ge::DataType dtype, ge::Format format = ge::FORMAT_ND) {
  ge::Shape shape(dims);
  ge::TensorDesc desc(shape, format, dtype);
  ge::Tensor tensor(desc, reinterpret_cast<uint8_t*>(vals.data()), vals.size() * sizeof(T));
  return tensor;
}

bool FindAttrValue(const domi::tensorflow::NodeDef *node_def, const string attr_name,
                                       domi::tensorflow::AttrValue &attr_value) {
    if (node_def == nullptr) {
        return false;
    }
    const google::protobuf::Map<std::string, domi::tensorflow::AttrValue> &attr = node_def->attr();
    const google::protobuf::Map<std::string, domi::tensorflow::AttrValue>::const_iterator it = attr.find(attr_name);
    if (it != attr.end()) {
        attr_value = it->second;
        return true;
    }
    return false;
}

Status ParseParamsClipByValueOp(const Message *op_src, ge::Operator& op_dest) {
    const domi::tensorflow::NodeDef *const node_src = ge::PtrToPtr<const ascend_private::protobuf::Message,
                                                      const domi::tensorflow::NodeDef>(op_src);
    int n = node_src->input_size();
    auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
    std::cout << "dynamic input: " << n << std::endl << std::flush;
    opDesc->AddDynamicInputDesc("x", n);
    opDesc->AddDynamicOutputDesc("y", 1);
    std::cout << "before node->attribute()" << std::endl << std::flush;
    
    domi::tensorflow::AttrValue clip_value_min_value;
    domi::tensorflow::AttrValue clip_value_max_value;
    /*
    FindAttrValue(node_src, "clip_value_min", clip_value_min_value);
    FindAttrValue(node_src, "clip_value_max", clip_value_max_value);
    domi::tensorflow::AttrValue_ListValue list_clip_value_min_values;
    domi::tensorflow::AttrValue_ListValue list_clip_value_max_values;
    list_clip_value_min_values = clip_value_min_value.list();
    list_clip_value_max_values = clip_value_max_value.list();
    int size_min = list_clip_value_min_values.i_size();
    int size_max = list_clip_value_max_values.i_size();
    std::cout << "size_min: " << size_min << " size_max: " << size_max << std::endl << std::flush;
    float clip_value_min = list_clip_value_min_values.i(0);
    float clip_value_max = list_clip_value_max_values.i(0);
    */
    /*FindAttrValue(node_src, "clip_value_min", clip_value_min_value);
    if (clip_value_min_value.has_list()) {
        int32_t dynamic_tensor_num = clip_value_min_value.list().type_size();
        std::cout << "in has_list" << std::endl << std::flush;
    } else {
        float clip_value_min = static_cast<float>(clip_value_min_value.i());
        std::cout << "in pure num" << std::endl << std::flush;
        std::cout << "clip_value_min: " << clip_value_min << std::endl << std::flush;
    }
    float clip_value_min = 5.0;
    float clip_value_max = 12.0;

    op_dest.SetAttr("clip_value_min", clip_value_min);
    op_dest.SetAttr("clip_value_max", clip_value_max);
    */
    if (AutoMappingFn(op_src, op_dest) != SUCCESS) {
        return FAILED;
    }
    op_dest.SetAttr("name", node_src->name());
    op_dest.SetAttr("original_type", "ClipByValueOp");
    // op_dest.SetAttr("original_type", "ClipByValue");
    return SUCCESS;
}

static Status ParseOpToGraphClipByValueOp(const ge::Operator &op, ge::Graph &graph) {
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        // OP_LOGE(op.GetName().c_str(), "get name from op failed");
        return FAILED;
    }
    
    ge::Operator data_0 = op::Data("input_data").set_attr_index(0);
    // ge::Tensor value_min;
    // ge::Tensor value_max;
    
    auto all_attr = op.GetAllAttrNamesAndTypes(); 
    float value_min;
    float value_max;
    if (op.GetAttr("clip_value_min", value_min) != SUCCESS) {
        // OP_LOGE(TbeGetName(op).c_str(), "get clip_value_min from op failed");
        std::cout << "value min: " << value_min << std::endl << std::flush;
        return FAILED;
    }
    if (op.GetAttr("clip_value_max", value_max) != SUCCESS) {
        // OP_LOGE(TbeGetName(op).c_str(), "get clip_value_max from op failed");
        std::cout << "value max: " << value_max << std::endl << std::flush;
        return FAILED;
    }
    // value_min_float16 = static_cast<Eigen::half>(value_min);
    // value_max_float16 = static_cast<Eigen::half>(value_max);
    value_min = static_cast<float>(value_min);
    value_max = static_cast<float>(value_max);
    // std::vector<Eigen::half> value_min_vector = {value_min_float16};
    std::vector<float> value_min_vector = {value_min};
    std::vector<int64_t> value_min_len = {1};
    // std::vector<Eigen::half> value_max_vector = {value_max_float16};
    std::vector<float> value_max_vector = {value_max};
    std::vector<int64_t> value_max_len = {1};
    std::cout << "after init max min len " <<  std::endl << std::flush;
    // ge::Tensor value_min_tensor = Vec2Tensor(value_min_vector, value_min_len, DT_FLOAT16);
    // ge::Tensor value_max_tensor = Vec2Tensor(value_max_vector, value_max_len, DT_FLOAT16);
    ge::Tensor value_min_tensor = Vec2Tensor(value_min_vector, value_min_len, DT_FLOAT);
    ge::Tensor value_max_tensor = Vec2Tensor(value_max_vector, value_max_len, DT_FLOAT);

    std::cout << "after Vec2Tensor " <<  std::endl << std::flush;
    auto const_value_min = op::Const("clip_value_min").set_attr_value(value_min_tensor);
    auto const_value_max = op::Const("clip_value_max").set_attr_value(value_max_tensor);

    std::cout << "after get const operator " <<  std::endl << std::flush;
    auto clip_by_value = op::ClipByValue(ori_name)
                         .set_input_x(data_0)
                         .set_input_clip_value_min(const_value_min)
                         .set_input_clip_value_max(const_value_max);
    
    std::cout << "after constr graph" <<  std::endl << std::flush;
    std::vector<ge::Operator> inputs{data_0};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
    output_indexs.emplace_back(clip_by_value, vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(output_indexs);
    std::cout << "leave ParseOpToGraphClipByValueOp" <<  std::endl << std::flush;
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ClipByValueOp")
    // .OriginOpType("ClipByValue")
    .ParseParamsFn(ParseParamsClipByValueOp)
    .ParseOpToGraphFn(ParseOpToGraphClipByValueOp)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
