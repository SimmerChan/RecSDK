/* Copyright (C) 2020-2021. Huawei Technologies Co., Ltd. All
rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <iostream>
#include <string>
#include <vector>
#include "register/register.h"
#include "graph/operator.h"
#include "graph/graph.h"
#include "graph/operator_factory.h"

namespace domi {
using namespace ge;

static Status AddOptionalPlaceholderForRS(const ge::Operator &tf_op, ge::Graph &graph) {
    // 1. 创建一个RmaSwapMultiTables算子
    ge::AscendString op_name;
    tf_op.GetName(op_name);
    auto npu_fa_op = OperatorFactory::CreateOperator(op_name.GetString(), "RmaSwapMultiTables");
    // 2. 将ir中的属性映射到新算子上
    int table_num = 1;
    (void)tf_op.GetAttr("table_num", table_num);
    (void)npu_fa_op.SetAttr("table_num", table_num);

    string shm_swap_in = "";
    (void)tf_op.GetAttr("shm_swap_in", shm_swap_in);
    (void)npu_fa_op.SetAttr("shm_swap_in", shm_swap_in);

    string shm_swap_out = "";
    (void)tf_op.GetAttr("shm_swap_out", shm_swap_out);
    (void)npu_fa_op.SetAttr("shm_swap_out", shm_swap_out);

    // 3. 创建输入Data
    std::vector<Operator> inputs;
    for (size_t i = 0UL; i < tf_op.GetInputsSize(); i++) {
        const std::string data_name = "Data_" + std::to_string(i);
        Operator data_op = OperatorFactory::CreateOperator(data_name.c_str(), "Data");
        (void)data_op.SetAttr("index", static_cast<int32_t>(i));
        inputs.emplace_back(data_op);
    }

    size_t index = 0UL;
    //4. 必选输入直接设置Data到算子输入
    (void)npu_fa_op.SetInput("swap_in_index", inputs[index++]);

    (void)npu_fa_op.SetInput("swap_out_index", inputs[index++]);

    (void)npu_fa_op.SetInput("table_a", inputs[index++]);

    // 5. 可选输入需要判断type属性的个数是否为0，不为0则表示optionalInput已经使能
    std::vector<DataType> table1_type;
    (void)tf_op.GetAttr("table_b_type", table1_type);
    if (!table1_type.empty()) {
        (void)npu_fa_op.SetInput("table_b", inputs[index++]);
    }

    std::vector<DataType> table2_type;
    (void)tf_op.GetAttr("table_c_type", table2_type);
    if (!table2_type.empty()) {
        (void)npu_fa_op.SetInput("table_c", inputs[index++]);
    }

    std::vector<DataType> table3_type;
    (void)tf_op.GetAttr("table_d_type", table3_type);
    if (!table3_type.empty()) {
        (void)npu_fa_op.SetInput("table_d", inputs[index++]);
    }

    std::vector<DataType> table4_type;
    (void)tf_op.GetAttr("table_e_type", table4_type);
    if (!table4_type.empty()) {
        (void)npu_fa_op.SetInput("table_e", inputs[index++]);
    }

    std::vector<DataType> table5_type;
    (void)tf_op.GetAttr("table_f_type", table5_type);
    if (!table5_type.empty()) {
        (void)npu_fa_op.SetInput("table_f", inputs[index++]);
    }

    std::cout << "npu_fa_op.GetOutputsSize() = " << npu_fa_op.GetOutputsSize();
    // 6. 使用FA算子的输出构造图的输出。
    std::vector<std::pair<Operator, std::vector<size_t>>> output_indexs;
    std::vector<size_t> node_output_index;
    for (size_t i = 0UL; i < npu_fa_op.GetOutputsSize(); i++) {
        node_output_index.emplace_back(i);
    }
    (void)output_indexs.emplace_back(std::make_pair(npu_fa_op, node_output_index));
    (void)graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}

static Status RmaSwapMultiTablesMapping(const ge::Operator& op_src, ge::Operator& op_dst) {
    // 1. 调用默认映射函数即可
    if (AutoMappingByOpFn(op_src, op_dst) != ge::GRAPH_SUCCESS) {
        return FAILED;
    }
    // 2. 需要将tf框架中的原算子名字设置到GE算子的original_type属性中，为触发后续调整optionInput的连边的动作使用
    op_dst.SetAttr("original_type", "RmaSwapMultiTables");
    return SUCCESS;
}

REGISTER_CUSTOM_OP("RmaSwapMultiTables")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RmaSwapMultiTables")
    .ParseParamsByOperatorFn(RmaSwapMultiTablesMapping) // 注册此函数用于实现算子本身属性的映射
    .ParseOpToGraphFn(AddOptionalPlaceholderForRS) // 注册此函数用于实现将tf中的输入转化为可选输入，改变连边关系
    .ImplyType(ImplyType::TVM);
}  // namespace domi
