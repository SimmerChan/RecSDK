/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

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
// Copyright 2021 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tf_bridge/utils/common_utils.h"

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/graph/graph.h"
#include "tf_bridge/tf_compatible.h"

namespace tensorflow {
namespace npu_xla {
namespace util {

bool HasOpType(const Graph& graph, absl::string_view op_type)
{
    for (auto n : graph.op_nodes()) {
        if (n->type_string() == op_type) {
            return true;
        }
    }
    return false;
}

// 添加函数到工作列表
void AddFunctionToWorklist(const string& func_name, const FunctionLibraryDefinition& flib,
                           std::unordered_set<string>& keep_funcs, std::vector<const FunctionDef*>& worklist)
{
    const FunctionDef* func = flib.Find(func_name);
    if (func && keep_funcs.find(func_name) == keep_funcs.end()) {
        worklist.push_back(func);
    }
}

// 处理节点中的函数引用
void ProcessNodeFunctionReferences(const NodeDef& node, const FunctionLibraryDefinition& flib,
                                   std::unordered_set<string>& keep_funcs, std::vector<const FunctionDef*>& worklist)
{
    // 节点本身可能是函数调用
    AddFunctionToWorklist(node.op(), flib, keep_funcs, worklist);

    // 节点可能有引用函数的属性
    for (const auto& attr : node.attr()) {
        const auto& attr_value = attr.second;

        // 1. AttrValue.func
        if (attr_value.has_func()) {
            AddFunctionToWorklist(attr_value.func().name(), flib, keep_funcs, worklist);
        }

        // 2. AttrValue.ListValue.func
        if (attr_value.has_list()) {
            for (const auto& func : attr_value.list().func()) {
                AddFunctionToWorklist(func.name(), flib, keep_funcs, worklist);
            }
        }
    }
}

// 构建可达函数的库
FunctionDefLibrary BuildReachableFunctionLibrary(const std::unordered_set<string>& keep_funcs,
                                                 const FunctionLibraryDefinition& flib)
{
    FunctionDefLibrary lib;
    for (const string& func_name : keep_funcs) {
        const FunctionDef* func = CHECK_NOTNULL(flib.Find(func_name));
        *lib.add_function() = *func;

        const string grad_func_name = flib.FindGradient(func_name);
        if (!grad_func_name.empty()) {
            GradientDef* gd = lib.add_gradient();
            gd->set_function_name(func_name);
            gd->set_gradient_func(grad_func_name);
        }
    }
    return lib;
}

std::unique_ptr<FunctionLibraryDefinition> ReachableDefinitions(const FunctionLibraryDefinition& flib,
                                                                const FunctionDef& entry_func)
{
    // 可从优化图访问的函数
    std::unordered_set<string> keep_funcs;

    // 插入入口函数本身
    keep_funcs.insert(entry_func.signature().name());

    std::vector<const FunctionDef*> worklist;
    worklist.reserve(flib.num_functions());

    // 处理入口函数中的所有节点
    const auto& graph_nodes = entry_func.node_def();
    for (const auto& node : graph_nodes) {
        ProcessNodeFunctionReferences(node, flib, keep_funcs, worklist);
    }

    // 处理所有可达函数
    while (!worklist.empty()) {
        const FunctionDef* func = worklist.back();
        worklist.pop_back();

        const string& func_name = func->signature().name();
        keep_funcs.insert(func_name);

        // 查找函数体中调用的所有函数
        const auto& func_body = func->node_def();
        for (const auto& node : func_body) {
            ProcessNodeFunctionReferences(node, flib, keep_funcs, worklist);
        }

        // 检查函数是否有注册的梯度
        const string grad_func_name = flib.FindGradient(func_name);
        if (!grad_func_name.empty()) {
            AddFunctionToWorklist(grad_func_name, flib, keep_funcs, worklist);
        }
    }

    // 构建可达函数的库
    FunctionDefLibrary lib = BuildReachableFunctionLibrary(keep_funcs, flib);

    // 使用std::make_unique替代new
    return std::make_unique<FunctionLibraryDefinition>(flib.default_registry(), std::move(lib));
}

}  // namespace util
}  // namespace npu_xla
}  // namespace tensorflow
