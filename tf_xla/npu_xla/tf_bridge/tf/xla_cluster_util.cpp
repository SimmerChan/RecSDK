/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tf_bridge/tf/xla_cluster_util.h"

#include <unordered_map>
#include <unordered_set>

#include "absl/container/inlined_vector.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/env_var.h"
#include "tf_bridge/common.h"
#include "tf_bridge/tf/errors.h"
#include "tf_bridge/tf/flags.h"
#include "tf_bridge/tf/xla_config_registry.h"
#include "tf_bridge/tf_compatible.h"

namespace tensorflow {
namespace npu_xla {

const char* const K_XLA_CLUSTER_ATTR = "_XlaCluster";
const char* const K_XLA_OUTSIDE_COMPILATION_ATTR = "_XlaOutsideCompilation";
const char* const K_XLA_COMPILE_TIME_CONSTANT_INPUTS_ATTR = "_XlaCompileTimeConstantInputs";
const char* const K_MLIR_COMPILE_TIME_CONSTANT_INPUTS_ATTR = "_MlirCompileTimeConstantInputs";
const char* const K_MLIR_COMPILE_TIME_FIXED_SHAPE_INPUTS_ATTR = "_MlirCompileTimeFixedShapeInputs";

namespace {
// Returns a string describing how an edge from src to dst would
// create a cycle.
string DescribeCycle(const GraphCycles* cycles, const Graph& graph, int src, int dst)
{
    int32 max_path_size = graph.num_node_ids() + 1;
    std::vector<int32> path(max_path_size);
    int32 path_size = cycles->FindPath(dst, src, max_path_size, path.data());
    if (path_size == 0) {
        return "";
    }

    auto nodeName = [&graph](int nodeId) {
        if (!FastBoundsCheck(nodeId, graph.num_node_ids())) {
            return string("(null)");
        }
        auto* node = graph.FindNodeId(nodeId);
        if (node == nullptr) {
            return string("(null)");
        }
        return node->name();
    };

    string description;
    absl::StrAppend(&description, "Edge from ", nodeName(src), " to ", nodeName(dst), " would create a cycle.\n");
    path.resize(path_size);
    for (int32 nodeId : path) {
        string ascii_art;
        if (nodeId == dst) {
            ascii_art = "+-> ";
        } else if (nodeId != src) {
            ascii_art = "|   ";
        } else {
            ascii_art = "+-- ";
        }
        absl::StrAppend(&description, ascii_art, nodeName(nodeId), "\n");
    }
    return description;
}

bool AlwaysForwardsRefInput(const Node& node)
{
    return node.IsIdentity();
}

}  // namespace

bool HasForwardedRefInput(const Node& node)
{
    if (AlwaysForwardsRefInput(node)) {
        for (const Edge* incoming_edge : node.in_edges()) {
            if (incoming_edge->IsControlEdge()) {
                continue;
            }

            Node* incoming_node = incoming_edge->src();
            if (IsRefType(incoming_node->output_type(incoming_edge->src_output()))) {
                VLOG(VLOG_LEVEL_2) << "Node " << node.def().ShortDebugString() << " has ref input "
                                   << incoming_node->name() << " " << incoming_node->type_string();
                return true;
            }
        }
    }
    return false;
}

absl::StatusOr<bool> CreateCycleDetectionGraph(const Graph* graph, GraphCycles* cycles)
{
    for (int i = 0; i < graph->num_node_ids(); ++i) {
        // We rely on the node IDs in the cycle detection graph being consecutive
        // integers starting from 0.
        CHECK_EQ(i, cycles->NewNode());
    }

    // Compute the loop structure of the graph.
    std::vector<ControlFlowInfo> control_flow_info;
    TF_RETURN_IF_ERROR(BuildControlFlowInfo(graph, &control_flow_info));

    // The clustering code must avoid adding cycles to the graph to prevent
    // deadlock. However, the graph may contain loops, which would trigger the
    // cycle detection code. To handle loops, we alter the structure of the cycle
    // detection graph, disconnecting each loop from the enclosing graph.
    // Specifically, we:
    // * add a new "frame" node for each loop.
    // * replace edges to "Enter" nodes, and edges from "Exit" nodes with edges
    //   to/from the corresponding frame node. In essence, we collapse the loop
    //   into a single node for the purpose of cycle detection in the enclosing
    //   graph.
    // * the body of the loop should now be disconnected from the rest of the
    //   graph; we make it acyclic by breaking loop backedges (edges outgoing from
    //   "NextIteration" nodes.

    // Map from frame name strings to node IDs in the cycle detection graph.
    std::unordered_map<string, int> frame_nodes;

    // Get the cycle graph node ID for frame 'frame_name', or add one if none
    // exists.
    auto getOrAddFrameNodeId = [&frame_nodes, cycles](const string& frame_name) {
        int& frameId = frame_nodes.emplace(frame_name, -1).first->second;
        if (frameId < 0) {
            // The emplace succeeded; we have not allocated a frame node yet.
            frameId = cycles->NewNode();
        }
        return frameId;
    };

    for (Edge const * edge : graph->edges()) {
        if (edge->dst()->IsEnter() || edge->src()->IsExit()) {
            const char* src_type = "pre-enter";
            const char* dst_type = "post-exit";
            int src = edge->src()->id();
            int dst = edge->dst()->id();

            if (edge->dst()->IsEnter()) {
                // Lift edges to an "Enter" node to the corresponding frame node.
                const string& frame_name = control_flow_info[edge->dst()->id()].frame_name;
                dst = getOrAddFrameNodeId(frame_name);
                dst_type = "frame";
            }

            if (edge->src()->IsExit()) {
                // Lift edges from an "Exit" node to the corresponding frame node.
                const string& frame_name = control_flow_info[edge->src()->id()].frame_name;
                src = getOrAddFrameNodeId(frame_name);
                src_type = "frame";
            }

            if (!cycles->InsertEdge(src, dst)) {
                // sophisticated SCC based algorithm, but for now we bail out.
                VLOG(VLOG_LEVEL_1) << "Cycle detected when adding " << src_type << "->" << dst_type
                                   << " edge: " << DescribeCycle(cycles, *graph, src, dst);
                return false;
            }
            // Drop the original edge.
            continue;
        }
        if (edge->src()->IsNextIteration()) {
            // Break loop back-edges.
            continue;
        }
        if (!cycles->InsertEdge(edge->src()->id(), edge->dst()->id())) {
            // This should never happen. All cycles in the graph should contain
            // a control flow operator.
            return errors::Internal("Found cycle in graph without control flow operator during XLA "
                                    "compilation: ",
                                    DescribeCycle(cycles, *graph, edge->src()->id(), edge->dst()->id()));
        }
    }

    return true;
}

absl::optional<absl::string_view> GetXlaClusterForNode(const Node& node)
{
    const AttrValue* attr_value = node.attrs().Find(K_XLA_CLUSTER_ATTR);
    if (attr_value == nullptr) {
        return absl::nullopt;
    }
    Status s = AttrValueHasType(*attr_value, "string");
    if (!s.ok()) {
        return absl::nullopt;
    }
    return attr_value->s();
}

bool HasResourceInputOrOutput(const Node& node)
{
    return std::find(node.input_types().begin(), node.input_types().end(), DT_RESOURCE) != node.input_types().end() ||
           std::find(node.output_types().begin(), node.output_types().end(), DT_RESOURCE) != node.output_types().end();
}

void RemoveFromXlaCluster(NodeDef* node_def)
{
    node_def->mutable_attr()->erase(K_XLA_CLUSTER_ATTR);
}

void RemoveFromXlaCluster(Node* node)
{
    node->ClearAttr(K_XLA_CLUSTER_ATTR);
}

namespace {
using XlaGlobalJitLevel = xla_config_registry::XlaGlobalJitLevel;

XlaGlobalJitLevel GetXlaGlobalJitLevel(const OptimizerOptions::GlobalJitLevel& jit_level_in_session_opts)
{
    XlaGlobalJitLevel result;

    if (jit_level_in_session_opts == OptimizerOptions::DEFAULT) {
        // To set compilation to be on by default, change the following line.
        result.single_gpu = result.general = OptimizerOptions::OFF;
    } else {
        result.single_gpu = result.general = jit_level_in_session_opts;
    }

    // If the flag tf_xla_auto_jit is a valid, non-DEFAULT setting, it overrides
    // the setting in ConfigProto.
    MarkForCompilationPassFlags* flags = GetMarkForCompilationPassFlags();
    if (flags->xla_auto_jit_flag.optimization_level_single_gpu != OptimizerOptions::DEFAULT) {
        result.single_gpu =
            static_cast<OptimizerOptions::GlobalJitLevel>(flags->xla_auto_jit_flag.optimization_level_single_gpu);
    }
    if (flags->xla_auto_jit_flag.optimization_level_general != OptimizerOptions::DEFAULT) {
        result.general =
            static_cast<OptimizerOptions::GlobalJitLevel>(flags->xla_auto_jit_flag.optimization_level_general);
    }

    return result;
}

int GetGpuNumber(const string& device_name)
{
    DeviceNameUtils::ParsedName parsed_name;
    if (!DeviceNameUtils::ParseFullName(device_name, &parsed_name)) {
        return -1;
    }

    return parsed_name.type == DEVICE_GPU ? parsed_name.id : -1;
}
}  // namespace

bool IsSingleGpuGraph(const Graph& g)
{
    int gpusSeen = 0;
    std::unordered_set<string> devicesSeen;

    for (Node* n : g.op_nodes()) {
        if (devicesSeen.find(n->assigned_device_name()) != devicesSeen.end()) {
            continue;
        }

        int gpuNumber = GetGpuNumber(n->assigned_device_name());
        if (gpuNumber != -1) {
            if (++gpus_seen > 1) {
                return false;
            }
        }

        devicesSeen.insert(n->assigned_device_name());
    }

    return gpusSeen == 1;
}

OptimizerOptions::GlobalJitLevel GetGlobalJitLevelForGraph(const GraphOptimizationPassOptions& options)
{
    OptimizerOptions::GlobalJitLevel jitLevelInSessionOpts =
        options.session_options->config.graph_options().optimizer_options().global_jit_level();
    // enable JIT when NPU_XLA is on
    if (GetTfBridgeOptions()->enable_npu_xla && jitLevelInSessionOpts == OptimizerOptions::DEFAULT) {
        jitLevelInSessionOpts = OptimizerOptions::ON_1;
    }

    XlaGlobalJitLevel xlaGlobalJitLevel = GetXlaGlobalJitLevel(jitLevelInSessionOpts);
    if (xlaGlobalJitLevel.single_gpu == xlaGlobalJitLevel.general) {
        return xlaGlobalJitLevel.single_gpu;
    }
    OptimizerOptions::GlobalJitLevel result =
        IsSingleGpuGraph(**options.graph) ? xla_global_jit_level.single_gpu : xla_global_jit_level.general;
    return result;
}

bool MayCallFunction(const Node& n, const FunctionLibraryDefinition* flib_def)
{
    if (flib_def->Contains(n.type_string())) {
        return true;
    }

    // This is a conservative check: there may be nodes with a `func`
    // attribute that do not make function calls.
    return std::any_of(
        n.def().attr().begin(), n.def().attr().end(),
        [](const std::pair<string, AttrValue>& name_attr_pair) { return name_attr_pair.second.has_func(); });
}
bool IsShapeConsumerOp(const Node& node)
{
    return node.type_string() == "Shape" || node.type_string() == "Rank" || node.type_string() == "Size";
}

namespace {
struct ClusterInfo {
    int size;

    // Maps op names to the number of times they appear in the cluster.
    std::unordered_map<string, int> op_histogram;
};

void HistogramMapToRepeatedOpAndCount(protobuf::RepeatedPtrField<XlaAutoClusteringSummary::OpAndCount>* result,
                                      const std::unordered_map<string, int>& histogram)
{
    for (const auto& pair : histogram) {
        XlaAutoClusteringSummary::OpAndCount* new_entry = result->Add();
        new_entry->set_op(std::string(pair.first));
        new_entry->set_count(pair.second);
    }

    std::sort(result->begin(), result->end(),
              [](const XlaAutoClusteringSummary::OpAndCount& a, const XlaAutoClusteringSummary::OpAndCount& b) {
                  return a.op() < b.op();
              });
}

void ClusterInfoToProtobuf(XlaAutoClusteringSummary::Cluster* result, absl::string_view name, const ClusterInfo& info)
{
    result->set_name(std::string(name));
    result->set_size(info.size);
    HistogramMapToRepeatedOpAndCount(result->mutable_op_histogram(), info.op_histogram);
}
}  // namespace

XlaAutoClusteringSummary GetXlaAutoClusteringSummary(const Graph& graph)
{
    std::unordered_map<string, ClusterInfo> cluster_name_to_info;
    XlaAutoClusteringSummary result;

    std::unordered_map<string, int> unclustered_op_histogram;

    for (Node* n : graph.nodes()) {
        absl::optional<absl::string_view> cluster_name = GetXlaClusterForNode(*n);
        if (cluster_name) {
            result.set_clustered_node_count(result.clustered_node_count() + 1);
            ClusterInfo* info = &cluster_name_to_info[std::string(*cluster_name)];
            info->size++;
            info->op_histogram[n->type_string()]++;
        } else {
            result.set_unclustered_node_count(result.unclustered_node_count() + 1);
            unclustered_op_histogram[n->type_string()]++;
        }
    }

    for (const auto& pair : cluster_name_to_info) {
        XlaAutoClusteringSummary::Cluster* new_cluster = result.add_clusters();
        ClusterInfoToProtobuf(new_cluster, pair.first, pair.second);
    }

    std::sort(result.mutable_clusters()->begin(), result.mutable_clusters()->end(),
              [&](const XlaAutoClusteringSummary::Cluster& a, const XlaAutoClusteringSummary::Cluster& b) {
                  return a.name() < b.name();
              });

    HistogramMapToRepeatedOpAndCount(result.mutable_unclustered_op_histogram(), unclustered_op_histogram);

    return result;
}

namespace {
const int INLINE_VEC_SIZE = 2;
using CallTargetListTy = absl::InlinedVector<NameAttrList, INLINE_VEC_SIZE>;

CallTargetListTy GetCallTargetListFromNode(const Node& n, FunctionLibraryRuntime* lib_runtime)
{
    const FunctionLibraryDefinition& flib_def = *lib_runtime->GetFunctionLibraryDefinition();
    if (flib_def.Find(n.type_string())) {
        NameAttrList callee;
        callee.set_name(n.type_string());
        *callee.mutable_attr() = n.def().attr();
        return {callee};
    }

    CallTargetListTy result;
    for (const auto& name_attr_pair : n.attrs()) {
        const AttrValue& attr_value = name_attr_pair.second;
        if (attr_value.value_case() == AttrValue::kFunc) {
            result.push_back(attr_value.func());
        } else if (attr_value.value_case() == AttrValue::kList) {
            result.insert(result.end(), attr_value.list().func().begin(), attr_value.list().func().end());
        }
    }

    return result;
}

enum class Direction {
    K_FORWARD,
    K_BACKWARD
};

Status GetNodesRelatedToRefVariablesInDirection(const Graph& graph, FunctionLibraryRuntime* lib_runtime,
                                                Direction direction, int depth, std::unordered_set<Node*>* result);

absl::StatusOr<bool> DoesAnyCalleeHaveRefNodes(const CallTargetListTy& call_target_list,
                                               FunctionLibraryRuntime* lib_runtime, Direction direction, int depth)
{
    const int kMaxDepth = 10;

    if (depth == kMaxDepth && !call_target_list.empty()) {
        // Conservative answer to avoid recursing too much.
        return true;
    }

    std::unordered_set<Node*> callee_ref_nodes;
    for (const NameAttrList& call_target : call_target_list) {
        const OpRegistrationData* op_reg;
        if (OpRegistry::Global()->LookUp(call_target.name(), &op_reg).ok()) {
            const OpDef& op = op_reg->op_def;
            if (std::any_of(op.output_arg().begin(), op.output_arg().end(),
                            [](const OpDef::ArgDef arg) { return arg.is_ref(); })) {
                return true;
            }
            continue;
        }

        callee_ref_nodes.clear();
        FunctionLibraryRuntime::Handle handle;
        if (!lib_runtime->Instantiate(call_target.name(), AttrSlice(&call_target.attr()), &handle).ok()) {
            VLOG(VLOG_LEVEL_2) << "Could not find " << call_target.name() << " in the function library.";
            // Since we don't know the semantic of `n` we don't know if this is an
            // error.  We return true to signal a conservative answer.
            return true;
        }

        auto release_handle_on_return = gtl::MakeCleanup([&] { TF_CHECK_OK(lib_runtime->ReleaseHandle(handle)); });

        const FunctionBody* fbody = lib_runtime->GetFunctionBody(handle);
        TF_RETURN_IF_ERROR(GetNodesRelatedToRefVariablesInDirection(*fbody->graph, lib_runtime, direction, depth + 1,
                                                                    &callee_ref_nodes));

        // We could possibly use something cheaper than
        // GetNodesRelatedToRefVariablesInDirection since we only care about the
        // size of `callee_ref_nodes` but for now we don't ceare.
        if (!callee_ref_nodes.empty()) {
            return true;
        }
    }

    return false;
}

// Helper for GetNodesRelatedToRefVariables that traverses the graph in one
// direction.
Status GetNodesRelatedToRefVariablesInDirection(const Graph& graph, FunctionLibraryRuntime* lib_runtime,
                                                Direction direction, int depth, std::unordered_set<Node*>* result)
{
    std::vector<Node*> nodes_in_order;
    if (direction == Direction::K_FORWARD) {
        GetReversePostOrder(graph, &nodes_in_order, NodeComparatorName());
    } else {
        GetPostOrder(graph, &nodes_in_order, NodeComparatorName());
    }

    size_t oldResultSize;
    int iterations = 0;

    const int kMaxIterations = 10 * 1000;

    std::vector<bool> callee_has_ref_nodes_cache;
    callee_has_ref_nodes_cache.resize(graph.num_node_ids());

    auto does_callee_have_ref_nodes = [&](Node* n) -> absl::StatusOr<bool> {
        if (iterations == 1) {
            TF_ASSIGN_OR_RETURN(
                bool callee_has_ref_nodes,
                DoesAnyCalleeHaveRefNodes(GetCallTargetListFromNode(*n, lib_runtime), lib_runtime, direction, depth));
            callee_has_ref_nodes_cache[n->id()] = callee_has_ref_nodes;
            return callee_has_ref_nodes;
        } else {
            return {callee_has_ref_nodes_cache[n->id()]};
        }
    };

    do {
        TF_RET_CHECK(iterations++ < kMaxIterations) << "infinite loop?";

        oldResultSize = result->size();
        for (Node* n : nodes_in_order) {
            if (n->IsSource() || n->IsSink()) {
                continue;
            }

            bool inserted_n = false;
            const EdgeSet& edges = direction == Direction::K_FORWARD ? n->in_edges() : n->out_edges();
            for (const Edge* e : edges) {
                if (result->find(direction == Direction::K_FORWARD ? e->src() : e->dst()) != result->end()) {
                    result->insert(n);
                    inserted_n = true;
                    break;
                }
            }

            if (inserted_n) {
                continue;
            }

            if (direction == Direction::K_FORWARD &&
                std::any_of(n->output_types().begin(), n->output_types().end(), IsRefType)) {
                result->insert(n);
                continue;
            }

            TF_ASSIGN_OR_RETURN(bool callee_has_ref_nodes, does_callee_have_ref_nodes(n));
            if (callee_has_ref_nodes) {
                result->insert(n);
                continue;
            }
        }

        // Loop until convergence.
    } while (result->size() != oldResultSize);

    VLOG(VLOG_LEVEL_2) << "# iterations = " << iterations;

    return absl::OkStatus();
}
}  // namespace

absl::StatusOr<std::unordered_set<Node*>> GetNodesRelatedToRefVariables(const Graph& graph,
                                                                        FunctionLibraryRuntime* lib_runtime)
{
    std::unordered_set<Node*> result;
    TF_RETURN_IF_ERROR(GetNodesRelatedToRefVariablesInDirection(graph, lib_runtime, Direction::K_FORWARD, 0, &result));
    TF_RETURN_IF_ERROR(GetNodesRelatedToRefVariablesInDirection(graph, lib_runtime, Direction::K_BACKWARD, 0, &result));

    VLOG(VLOG_LEVEL_1) << "GetNodesRelatedToRefVariables() found " << result.size() << " nodes";
    return result;
}

// Register a callback for querying XlaGlobalJitLevel.
REGISTER_XLA_CONFIG_GETTER(GetXlaGlobalJitLevel);

}  // namespace npu_xla
}  // namespace tensorflow
