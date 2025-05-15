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

#include "tf_bridge/passes/build_npu_xla_op_pass.h"

#include <string>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tf_bridge/passes/encapsulate_subgraphs_pass.h"
#include "tf_bridge/tf_compatible.h"
#include "tf_bridge/utils/dump_graph.h"

namespace tensorflow {
namespace npu_xla {

namespace {

struct XlaClusterInfo {
    std::vector<NodeBuilder::NodeOut> constant_inputs;
    std::vector<NodeBuilder::NodeOut> fixed_shape_inputs;
    std::vector<NodeBuilder::NodeOut> non_const_or_fixedshape_host_inputs;
    std::vector<NodeBuilder::NodeOut> non_const_or_fixedshape_device_inputs;
    std::vector<NodeBuilder::NodeOut> resource_inputs;
    DataTypeVector host_rets;
    DataTypeVector device_rets;
    NameAttrList function;
};

NodeBuilder::NodeOut IncomingEdgeAsOutput(const Edge* e)
{
    return NodeBuilder::NodeOut(e->src(), e->src_output());
}

Status GetXlaClusterInfo(Node* n, XlaClusterInfo* result)
{
    int num_constant_inputs = 0;
    int num_fixed_shape_inputs = 0;
    int num_non_const_or_fixedshape_host_inputs = 0;
    int num_resource_inputs = 0;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), kXlaNumConstantArgsAttr, &num_constant_inputs));
    TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), kMlirNumFixedShapeArgsAttr, &num_fixed_shape_inputs));
    TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), kMlirNumHostArgsAttr, &num_non_const_or_fixedshape_host_inputs));
    TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), kXlaNumResourceArgsAttr, &num_resource_inputs));

    int num_non_const_or_fixedshape_device_inputs = n->num_inputs() - num_constant_inputs - num_fixed_shape_inputs -
                                                    num_non_const_or_fixedshape_host_inputs - num_resource_inputs;

    if (num_constant_inputs < 0 || num_resource_inputs < 0 || num_non_const_or_fixedshape_host_inputs < 0 ||
        num_non_const_or_fixedshape_device_inputs < 0) {
        return errors::InvalidArgument("Invalid number of constant/fixedshape/resource arguments to XLA "
                                       "kernel.");
    }

    std::vector<const Edge*> input_edges_vector;
    TF_RETURN_IF_ERROR(n->input_edges(&input_edges_vector));
    int idx = 0;
    for (const Edge* e : input_edges_vector) {
        if (idx < num_constant_inputs) {
            result->constant_inputs.push_back(IncomingEdgeAsOutput(e));
        } else if (idx < num_constant_inputs + num_fixed_shape_inputs) {
            result->fixed_shape_inputs.push_back(IncomingEdgeAsOutput(e));
        } else if (idx < num_constant_inputs + num_fixed_shape_inputs + num_non_const_or_fixedshape_host_inputs) {
            result->non_const_or_fixedshape_host_inputs.push_back(IncomingEdgeAsOutput(e));
        } else if (idx < num_constant_inputs + num_fixed_shape_inputs + num_non_const_or_fixedshape_host_inputs +
                             num_non_const_or_fixedshape_device_inputs) {
            result->non_const_or_fixedshape_device_inputs.push_back(IncomingEdgeAsOutput(e));
        } else {
            result->resource_inputs.push_back(IncomingEdgeAsOutput(e));
        }
        ++idx;
    }

    result->function.set_name(n->type_string());
    *result->function.mutable_attr() = n->def().attr();

    int num_host_rets = 0;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), kMlirNumHostRetsAttr, &num_host_rets));
    for (int i = 0; i < n->num_outputs(); ++i) {
        auto t = n->output_type(i);
        if (i < num_host_rets) {
            result->host_rets.push_back(t);
        } else {
            result->device_rets.push_back(t);
        }
    }

    return absl::OkStatus();
}

Status CopyIncomingControlEdges(Graph* g, Node* from, Node* to)
{
    for (const Edge* e : from->in_edges()) {
        if (e->IsControlEdge()) {
            g->AddControlEdge(e->src(), to);
        }
    }

    return absl::OkStatus();
}

void MoveOutgoingEdges(Graph* g, Node* old_node, Node* new_node)
{
    std::vector<const Edge*> out_edges(old_node->out_edges().begin(), old_node->out_edges().end());
    for (const Edge* edge : out_edges) {
        // TODO(sanjoy): This does not update NodeDef inputs.  To be able to update
        // NodeDef inputs we first need to fix encapsulate_subgraphs_pass to fix up
        // the NodeDef inputs to the function call nodes.
        g->AddEdge(new_node, edge->src_output(), edge->dst(), edge->dst_input());
        VLOG(1) << new_node->name() << ":" << edge->src_output() << " -> " << edge->dst()->name() << ":"
                << edge->dst_input();
        g->RemoveEdge(edge);
    }
}

Status CreateFallbackFunction(const GraphOptimizationPassOptions& options, const Node* node,
                              const NameAttrList& tf_func, NameAttrList* fallback_function)
{
    FunctionLibraryDefinition* const library = options.flib_def;
    string defunct_name = tf_func.name() + kDefunctionalizedSuffix;
    if (library->Find(defunct_name) != nullptr) {
        fallback_function->set_name(defunct_name);
        *fallback_function->mutable_attr() = node->def().attr();
    }
    return absl::OkStatus();
}

Status ReplaceNodeWithNpuXlaLaunchOp(const GraphOptimizationPassOptions& options, Graph* g, Node* n, bool inner)
{
    VLOG(1) << "Run ReplaceNodeWithNpuXlaLaunchOp with " << n->name();
    XlaClusterInfo cluster_info;
    TF_RETURN_IF_ERROR(GetXlaClusterInfo(n, &cluster_info));

    auto mlir_func = absl::make_unique<NameAttrList>();

    NameAttrList fallback_function;
    TF_RETURN_IF_ERROR(CreateFallbackFunction(options, n, cluster_info.function, &fallback_function));

    VLOG(1) << "is_mlir: " << inner;
    VLOG(1) << "const_input size: " << cluster_info.constant_inputs.size();
    VLOG(1) << "fixed_shape_input size: " << cluster_info.fixed_shape_inputs.size();
    VLOG(1) << "host args size: " << cluster_info.non_const_or_fixedshape_host_inputs.size();
    VLOG(1) << "host rets size: " << cluster_info.host_rets.size();

    NodeBuilder nb = NodeBuilder(n->name() + "_npu_xla_launch", "NpuXlaLaunch")
                         .Input(cluster_info.constant_inputs)
                         .Input(cluster_info.fixed_shape_inputs)
                         .Input(cluster_info.non_const_or_fixedshape_host_inputs)
                         .Input(cluster_info.non_const_or_fixedshape_device_inputs)
                         .Input(cluster_info.resource_inputs)
                         .Attr("Thostresults", cluster_info.host_rets)
                         .Attr("Tdeviceresults", cluster_info.device_rets)
                         .Attr("_XlaCompile", false)
                         // When not at top level clustering, mlir_func is
                         // always nullptr, the func_def is extracted by
                         // GetXlaClusterInfo and it's mlir function!
                         .Attr("mlir_function", cluster_info.function)
                         .Device(n->requested_device())
                         .AssignedDevice(n->assigned_device_name());

    Node* tao_op = nullptr;
    Status status = nb.Finalize(g, &tao_op);
    TF_CHECK_OK(status);

    TF_RETURN_IF_ERROR(CopyIncomingControlEdges(g, /*from=*/n, /*to=*/tao_op));

    MoveOutgoingEdges(g, /*old_node=*/n, /*new_node=*/tao_op);
    g->RemoveNode(n);
    return absl::OkStatus();
}

}  // namespace

Status BuildNpuXlaOpPass::Run(const GraphOptimizationPassOptions& options)
{
    VLOG(1) << "BuildNpuXlaOpPass::Run is called, inner XlaLaunch: " << inner_launch_;
    Graph* graph = options.graph->get();

    std::vector<Node*> target_nodes;
    for (Node* n : graph->op_nodes()) {
        // In all cases, only try to compile computational nodes.
        if (n->IsSend() || n->IsRecv() || n->IsControlFlow()) {
            continue;
        }
        if (IsXlaCompiledKernel(*n)) {
            target_nodes.push_back(n);
        }
    }
    for (auto n : target_nodes) {
        TF_RETURN_IF_ERROR(ReplaceNodeWithNpuXlaLaunchOp(options, graph, n, inner_launch_));
    }

    if (false) {
        VLOG(0) << dump_graph::DumpGraphToFile("build_tao_ops after", *graph, options.flib_def);
    }
    return absl::OkStatus();
}

}  // namespace npu_xla
}  // namespace tensorflow
