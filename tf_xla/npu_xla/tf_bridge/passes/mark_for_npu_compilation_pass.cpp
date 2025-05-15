/*
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
/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tf_bridge/passes/mark_for_npu_compilation_pass.h"

#include <algorithm>
#include <atomic>
#include <deque>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/placer.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/public/version.h"
#include "tf_bridge/common.h"
#include "tf_bridge/passes/defunctionalize_control_flow.h"
#include "tf_bridge/tf/compilability_check_util.h"
#include "tf_bridge/tf/const_analysis.h"
#include "tf_bridge/tf/deadness_analysis.h"
#include "tf_bridge/tf/defs.h"
#include "tf_bridge/tf/device_util.h"
#include "tf_bridge/tf/errors.h"
#include "tf_bridge/tf/flags.h"
#include "tf_bridge/tf/graphcycles.h"
#include "tf_bridge/tf/resource_operation_safety_analysis.h"
#include "tf_bridge/tf/resource_operation_table.h"
#include "tf_bridge/tf/union_find.h"
#include "tf_bridge/tf/util.h"
#include "tf_bridge/tf/xla_cluster_util.h"
#include "tf_bridge/tf/xla_op_registry.h"
#include "tf_bridge/tf_compatible.h"
#include "tf_bridge/utils/dump_graph.h"

namespace tensorflow {
namespace npu_xla {

namespace {
using DeadnessPredicate = DeadnessAnalysis::DeadnessPredicate;
using jit::DeviceId;
using jit::DeviceSet;

// The clusters we create here are eventually lowered into an
// _XlaCompile/_XlaRun pair with a TF executor "fallback" that uses the
// PartitionedCall op to execute the cluster in the regular graph executor if
// need be.  PartitionedCall, however, reruns the entire TF graph optimization
// pipeline over the cluster which includes this mark for compilation pass.  To
// avoid endlessly recursing we tag nodes that we've already visited with this
// attribute so that we can bail out if we see them a second time.
const char* K_XLA_ALREADY_CLUSTERED = "_XlaAlreadyClustered";

class MarkForCompilationPassImpl {
public:
    struct DebugOptions {
        // If true, do not respect the results of deadness analysis.
        bool ignoreDeadnessChecks;

        // If true, do not do safety checks to preserve TensorFlow's resource
        // variable concurrency semantics.
        bool ignoreResourceVariableChecks;

        // If true, do not respect the _XlaCompile=false attribute.
        bool ignoreXlaCompileAttr;

        int maxClusterSize;
        int minClusterSize;

        // Compiler fuel for the auto-clustering algorithm.
        //
        // We decrement this value by one on every time we choose a compilation
        // candidate and we stop clustering when it hits zero.  This means the
        // initial value for this variable (via --tf_xla_clustering_fuel=N)
        // effectively acts as a "cap" for how much we cluster and we can bisect
        // over this initial value to discover clustering decisions that cause a
        // miscompile or a performance regression.
        std::atomic<int64_t>* fuel;

        bool dumpGraphs;

        bool forceAllowTensorArrayOps;
        bool skipClusteredOps;

        bool clusterForMlir = true;

        // Override the `tf_xla_ops_to_cluster` if this field present.
        absl::optional<std::string> override_tf_xla_ops_to_cluster;

        std::string graphTag;
    };

    MarkForCompilationPassImpl(DebugOptions debugOptions, Graph* graph, FunctionLibraryDefinition* flib_def, Env* env,
                               OptimizerOptions::GlobalJitLevel global_jit_level)
        : debug_options_(debugOptions),
          graph_(graph),
          flib_def_(flib_def),
          env_(env),
          global_jit_level_(global_jit_level)
    {
    }

    Status Run();

private:
    // Represents a "cluster" or a connected subgraph of a TensorFlow graph.
    class Cluster {
    public:
        // Constructs a trivial cluster representing a single TF node.
        Cluster(int tfGraphNodeId, int effectiveClusterSize, bool hasFunctionalControlFlow, DeviceSet devices,
                std::optional<DeviceId> resource_op_device, std::optional<int> resource_var_operation_node_id,
                std::optional<DeadnessPredicate> deadness_predicate, bool isXlaCompileAttrTrue,
                std::optional<string> xla_scope)
            : cyclesGraphNodeId_(tfGraphNodeId),
              effectiveClusterSize_(effectiveClusterSize),
              hasFunctionalControlFlow_(hasFunctionalControlFlow),
              devices_(std::move(devices)),
              resource_op_device_(resource_op_device),
              deadness_predicate_(deadness_predicate),
              isXlaCompileAttrTrue_(isXlaCompileAttrTrue),
              xla_scope_(std::move(xla_scope))
        {
            if (resource_var_operation_node_id.has_value()) {
                resource_var_operation_node_ids_.push_back(*resource_var_operation_node_id);
            }
        }

        // Merges `other` into this cluster, and clears `other`.  This method is
        // closely tied with the implementation of `MarkForCompilationPassImpl`.
        void Merge(Cluster* other);

        // If this is a trivial cluster containing only one node then return the ID
        // of that node.  May not be called otherwise.
        int GetIdOfOnlyNode() const
        {
            DCHECK_EQ(ClusterSize(), 1);
            return CyclesGraphNodeId();
        }

        // The number of TF nodes in this cluster.
        int ClusterSize() const
        {
            return clusterSize_;
        }

        // The ID of the cluster as represented in `cycles_graph_`.
        int CyclesGraphNodeId() const
        {
            return cyclesGraphNodeId_;
        }

        // Sets the ID of the cluster as represented in `cycles_graph_`.
        void SetCyclesGraphNodeId(int cyclesGraphNodeId)
        {
            cyclesGraphNodeId_ = cyclesGraphNodeId;
        }

        // The size of the cluster excluding constant and identity nodes.
        int EffectiveClusterSize() const
        {
            return effectiveClusterSize_;
        }

        // True if the cluster has functional control flow like `If` and `While`.
        bool HasFunctionalControlFlow() const
        {
            return hasFunctionalControlFlow_;
        }

        // The set of devices nodes in the cluster are placed on.
        const DeviceSet& devices() const
        {
            return devices_;
        }

        // If the cluster has a resource operation then the device the resource
        // operation is placed on.  A cluster may have resource ops placed only on a
        // single device.
        const std::optional<DeviceId>& resource_op_device() const
        {
            return resource_op_device_;
        }

        // If not nullopt the a predicate that is true iff the cluster is alive.
        // Otherwise the user has (unsafely) disabled deadness analysis.  If this is
        // unset on a single Cluster instance then it is unset on all Cluster
        // instances.
        const std::optional<DeadnessPredicate>& deadness_predicate() const
        {
            return deadness_predicate_;
        }

        // If true then the cluster has a XlaCompile=true attribute on one of its
        // nodes.
        bool IsXlaCompileAttrTrue() const
        {
            return isXlaCompileAttrTrue_;
        }

        // If not nullopt then the all nodes in the cluster either do not have the
        // XlaScope attribute set or have it set to the value returned.
        const std::optional<string>& xla_scope() const
        {
            return xla_scope_;
        }

        // Returns the TF graph node IDs for the resource variable operations in
        // this cluster.
        absl::Span<const int> resource_var_operation_node_ids() const
        {
            return resource_var_operation_node_ids_;
        }

        string DebugString(const Graph& graph) const
        {
            Node* node = graph.FindNodeId(CyclesGraphNodeId());
            if (!node) {
                // This should never happen but we try to be resilient because this is a
                // debugging aid.
                return absl::StrCat("NULL NODE IN #", CyclesGraphNodeId());
            }

            if (ClusterSize() == 1) {
                return absl::StrCat("<", node->name(), " #", CyclesGraphNodeId(), ">");
            }

            return absl::StrCat("<", node->name(), " + ", ClusterSize() - 1, " others #", CyclesGraphNodeId(), ">");
        }

    private:
        int clusterSize_ = 1;
        int cyclesGraphNodeId_;
        int effectiveClusterSize_;
        bool hasFunctionalControlFlow_;
        DeviceSet devices_;
        std::optional<DeviceId> resource_op_device_;
        std::optional<DeadnessPredicate> deadness_predicate_;
        bool isXlaCompileAttrTrue_;
        std::optional<string> xla_scope_;
        std::vector<int> resource_var_operation_node_ids_;

        Cluster(const Cluster&) = delete;
        void operator=(const Cluster&) = delete;
    };

    // If `cluster` has only a single node then returns that, otherwise returns
    // nullptr.
    Node* GetOnlyNodeIn(const Cluster& cluster);

    // Returns true if `cluster` is a trivial cluster containing a "sink like"
    // node -- a NoOp node that only the Sink node control depends on.
    bool IsSinkLike(const Cluster& cluster);

    // Returns true if `cluster` looks like an "i++" operation on an integer
    // scalar resource variable.
    bool IsScalarIntegerResourceOperation(const Cluster& cluster);

    // ---------------------------------------------------------------------------
    // The pass proceeds in five steps, out of which `RunEdgeContractionLoop` and
    // `CreateClusters` do most of the heavy lifting.

    // Initializes some internal data structures.
    // If this returns false then Initialize exited early (either because there is
    // nothing to do or we saw a graph that we can't handle) and not all the
    // fields in this MarkForCompilationPassImpl instance are set up.
    absl::StatusOr<bool> Initialize();

    // Runs through the entire cluster graph in post-order and calls `fn(from,
    // to)` on each edge.  `fn(from, to)` is expected to return true if it was
    // able to contract `from`->`to`.
    //
    // Returns true if `fn` returned true for any edge.
    template <typename FnTy>
    absl::StatusOr<bool> ForEachEdgeInPostOrder(FnTy fn);

    // Contracts as many edges as possible to create XLA clusters.  After this
    // finishes the clustering decisions made are implicitly stored in
    // `clusters_`.
    Status RunEdgeContractionLoop();

    // Manifests the clustering decisions into the TF graph by tagging nodes with
    // an `_XlaCluster` attribute.  Also some basic filter logic, like
    // tf_xla_minClusterSize, are applied here.
    Status CreateClusters();

    Status DumpDebugInfo();

    bool IsCompilationCandidate(Node* n) const
    {
        return compilation_candidates_.find(n) != compilation_candidates_.end();
    }

    // Tries to contract the edge from cluster `from` to cluster `to`.  Returns
    // true if successful.
    absl::StatusOr<bool> TryToContractEdge(Cluster* from, Cluster* to);

    // Nodes that XLA can compile are put in `compilation_candidates_`.
    Status FindCompilationCandidates();

    bool CompilationDisallowedByXlaCompileAttr(Node* node);

    // Populates `clusters_`.
    Status BuildInitialClusterSet();

    absl::StatusOr<bool> ShouldCompileClusterImpl(const Cluster& cluster);

    absl::StatusOr<bool> ShouldCompileCluster(const Cluster& cluster);

    absl::StatusOr<bool> ClusteringWillIntroduceInterDeviceDependency(const Cluster& from, const Cluster& to);

    // Returns true if the devices in `cluster_a` and `cluster_b` are compatible
    // and therefore not a hindrance for combining the two clusters into a larger
    // cluster.
    absl::StatusOr<bool> AreDevicesCompatible(const Cluster& cluster_a, const Cluster& cluster_b);

    void DumpPostClusteringGraphs();
    void VLogClusteringSummary();

    Cluster* MakeNewCluster(int cyclesGraphNodeId, int effectiveClusterSize, bool hasFunctionalControlFlow,
                            const DeviceSet& device_set, std::optional<DeviceId> resource_op_device,
                            std::optional<int> resource_var_operation_node_id,
                            std::optional<DeadnessPredicate> deadness_predicate, bool isXlaCompileAttrTrue,
                            std::optional<string> xla_scope)
    {
        cluster_storage_.push_back(std::make_unique<Cluster>(
            cyclesGraphNodeId, effectiveClusterSize, hasFunctionalControlFlow, device_set, resource_op_device,
            resource_var_operation_node_id, deadness_predicate, isXlaCompileAttrTrue, xla_scope));
        return cluster_storage_.back().get();
    }

    std::optional<string> GetXlaScope(Node* n);

    // Returns the cluster for node `n`.  If two nodes, N1 and N2, are placed in
    // the same cluster by the clustering algorithm then this function will return
    // the same Cluster instance for N1 and N2.
    // Returns nullptr if `n` is not a compilation candidate.
    Cluster* GetClusterForNode(Node* n)
    {
        return cluster_for_node_[n->id()].Get();
    }

    // Returns the cluster for a node in `cycles_graph_`.  This uses the same
    // underlying map because of how we set things up, but we can do an additional
    // CHECK in this accessor.
    // Returns nullptr if `nodeId` is not a compilation candidate.
    Cluster* GetClusterForCyclesGraphNode(int nodeId)
    {
        // We have to check `graph_->FindNodeId(node) == nullptr` because we add all
        // nodes in [0, graph_->num_node_ids()) to the cycle detection graph but the
        // TF graph may be missing some node ids.
        if (nodeId >= graph_->num_node_ids() || graph_->FindNodeId(nodeId) == nullptr) {
            return nullptr;
        }
        Cluster* cluster = cluster_for_node_[nodeId].Get();
        if (cluster) {
            DCHECK_EQ(cluster->CyclesGraphNodeId(), nodeId);
        }
        return cluster;
    }

    bool LogNotContractableAndReturnFalse(Cluster* from, Cluster* to, absl::string_view reason);

    // Finds a path in `cycles_graph_` from `from` to `to` that is not a direct
    // edge from `from` to `to`.
    // Tries to find a path that contains at least one unclusterable node.
    std::vector<int> FindAlternatePathForDebugging(int from, int to);

    // Returns a string representing `cycles_graph_node_id`.  If the node is
    // unclusterable (either it is a phatom "frame" node or is not a compilation
    // candidate) then set `*found_unclustered` to true.
    string DebugStringForCyclesGraphNode(int node_id, bool* found_unclustered);

    // We could not contract the edge from `from` to `to`.  Return a string
    // describing an alternate path from `from` to `to` (besides the direct edge
    // from `from` to `to`) which would have created a cycle had we contracted the
    // edge.
    // Tries (if possible) to find a path that contains at least one unclusterable
    // node as it is surprising to the user if we print "A->B could not be
    // contracted because of the path [P,Q,R]" where P, Q and R are all clusters
    // since in that case a natural question is why we could not form a {A, P, Q,
    // R, B} cluster.
    string DescribePotentialCycle(int from, int to);

    // Merge the clusters `clusterFrom` and `clusterTo`. After this step the
    // larger combined cluster is represented by `clusterFrom`, but can have
    // `cycles_graph_`'s ID of either `clusterFrom` or `clusterTo` depending on
    // which way will require less operations.
    bool MergeClusters(Cluster* clusterFrom, Cluster* clusterTo)
    {
        int from = clusterFrom->CyclesGraphNodeId();
        int to = clusterTo->CyclesGraphNodeId();

        auto optional_merged_node = cycles_graph_.ContractEdge(from, to);
        if (!optional_merged_node) {
            VLOG(VLOG_LEVEL_2) << "Could not contract " << clusterFrom->DebugString(*graph_) << " -> "
                               << clusterTo->DebugString(*graph_)
                               << " because contracting the edge would create a cycle via "
                               << DescribePotentialCycle(from, to) << ".";
            return false;
        }

        // Merge the clusters.
        clusterFrom->Merge(clusterTo);

        // Merge the UnionFind<Cluster*>.
        cluster_for_node_[from].Merge(&cluster_for_node_[to]);

        return true;
    }

    string EdgeContractionFailureMsg(Cluster* from, Cluster* to, absl::string_view reason)
    {
        return absl::StrCat("Could not contract ", from->DebugString(*graph_), " -> ", to->DebugString(*graph_),
                            " because ", reason, ".");
    }

    DebugOptions debug_options_;
    Graph* graph_;
    FunctionLibraryDefinition* flib_def_;
    Env* env_;
    OptimizerOptions::GlobalJitLevel global_jit_level_;
    absl::flat_hash_map<const Cluster*, bool> should_compile_cluster_cache_;
    jit::DeviceInfoCache device_info_cache_;

    bool initialized_ = false;
    bool edgesContracted_ = false;
    bool clustersCreated_ = false;

    std::vector<std::unique_ptr<Cluster>> cluster_storage_;
    std::vector<UnionFind<Cluster*>> cluster_for_node_;
    GraphCycles cycles_graph_;
    OrderedNodeSet compilation_candidates_;
    std::unique_ptr<DeadnessAnalysis> deadness_analysis_;
    int64_t iterationCount_ = 0;
    absl::flat_hash_set<std::pair<int, int>> unsafe_resource_deps_;
};

std::vector<int> MarkForCompilationPassImpl::FindAlternatePathForDebugging(int from, int to)
{
    std::vector<int> rpo = cycles_graph_.AllNodesInPostOrder();
    absl::c_reverse(rpo);

    // best_pred_for_node[n] contains a predecessor of `n` that has an
    // unclusterable node in some path from `from` to itself.
    // best_pred_for_node[n] is unpopulated for nodes that are not reachable from
    // `from`.  We build this table up inductively by traversing the cycles graph
    // in RPO.
    absl::flat_hash_map<int, int> best_pred_for_node;
    best_pred_for_node[from] = -1;

    int rpoIndex = 0;
    int currentRpoNode;
    do {
        currentRpoNode = rpo[rpoIndex++];
        std::optional<int> some_pred;
        std::optional<int> preferred_pred;
        for (int pred : cycles_graph_.Predecessors(currentRpoNode)) {
            if (!best_pred_for_node.contains(pred)) {
                continue;
            }

            // Ignore the from->to edge since we're trying to find an alternate path.
            if (currentRpoNode == to && pred == from) {
                continue;
            }

            some_pred = pred;
            if (GetClusterForCyclesGraphNode(pred) == nullptr) {
                preferred_pred = pred;
            }
        }

        if (some_pred || preferred_pred) {
            best_pred_for_node[currentRpoNode] = preferred_pred.has_value() ? *preferred_pred : *some_pred;
        }
    } while (currentRpoNode != to);

    auto getBestPred = [&best_pred_for_node](int n) {
        auto it = best_pred_for_node.find(n);
        CHECK(it != best_pred_for_node.end());
        return it->second;
    };

    std::vector<int> path;
    int currentPathNode = getBestPred(to);
    while (currentPathNode != from) {
        path.push_back(currentPathNode);
        currentPathNode = getBestPred(currentPathNode);
    }

    absl::c_reverse(path);
    return path;
}

string MarkForCompilationPassImpl::DebugStringForCyclesGraphNode(int cycles_graph_node_id, bool* found_unclustered)
{
    Cluster* cluster = GetClusterForCyclesGraphNode(cycles_graph_node_id);
    if (cluster) {
        return cluster->DebugString(*graph_);
    }

    *found_unclustered = true;
    if (cycles_graph_node_id >= graph_->num_node_ids()) {
        return absl::StrCat("<oob #", cycles_graph_node_id, ">");
    }

    Node* node = graph_->FindNodeId(cycles_graph_node_id);
    if (!node) {
        return absl::StrCat("<bad #", cycles_graph_node_id, ">");
    }

    return node->name();
}

string MarkForCompilationPassImpl::DescribePotentialCycle(int from, int to)
{
    std::vector<string> path_str;
    bool foundUnclustered = false;
    absl::c_transform(FindAlternatePathForDebugging(from, to), std::back_inserter(path_str),
                      [&](int node_id) { return DebugStringForCyclesGraphNode(node_id, &foundUnclustered); });
    return absl::StrCat(!foundUnclustered ? "(all clusters) " : "", "[", absl::StrJoin(path_str, ","), "]");
}

void MarkForCompilationPassImpl::Cluster::Merge(Cluster* other)
{
    // We keep our own cyclesGraphNodeId_ to mirror what GraphCycles does.

    // Clearing out data structures in `other` is just a memory saving
    // optimization and not needed for correctness.

    clusterSize_ += other->clusterSize_;
    effectiveClusterSize_ += other->effectiveClusterSize_;
    hasFunctionalControlFlow_ |= other->hasFunctionalControlFlow_;

    devices_.UnionWith(other->devices_);

    DCHECK(!(resource_op_device_.has_value() && other->resource_op_device_.has_value()) ||
           *resource_op_device_ == *other->resource_op_device_)
        << "AreDevicesCompatible should have returned false otherwise!";

    if (!resource_op_device_.has_value()) {
        resource_op_device_ = other->resource_op_device_;
    }

    isXlaCompileAttrTrue_ |= other->isXlaCompileAttrTrue_;

    if (!xla_scope_.has_value()) {
        xla_scope_ = std::move(other->xla_scope_);
    }

    resource_var_operation_node_ids_.reserve(resource_var_operation_node_ids_.size() +
                                             other->resource_var_operation_node_ids_.size());
    absl::c_copy(other->resource_var_operation_node_ids_, std::back_inserter(resource_var_operation_node_ids_));
    other->resource_var_operation_node_ids_.clear();
}

Status IgnoreResourceOpForSafetyAnalysis(jit::DeviceInfoCache* device_info_cache, const Node& n, bool* ignore)
{
    // If a resource operation is assigned to XLA_CPU or XLA_GPU explicitly then
    // ignore it during resource operation safety analysis.  We need this hack
    // because of two reasons:
    //
    //  1. Operations assigned to XLA_CPU and XLA_GPU have to always be compiled.
    //  2. We don't support live-out values of type DT_RESOURCE and live-in values
    //     of type DT_RESOURCE that are not resource variables.
    //
    // Together these imply we cannot let resource variable safety analysis
    // constrain e.g. a TensorArrayV3->TensorArrayAssignV3 edge to be in different
    // clusters: both of them will have to be clustered because of (1) and we
    // won't be able to keep the edge between the two as neither the input to the
    // second XLA cluster nor the output from the first XLA cluster are supported
    // because of (2).
    if (n.assigned_device_name().empty()) {
        *ignore = false;
        return absl::OkStatus();
    }

    TF_ASSIGN_OR_RETURN(const XlaOpRegistry::DeviceRegistration* registration,
                        device_info_cache->GetCompilationDevice(n.assigned_device_name()));

    if (!registration) {
        *ignore = true;
    } else {
        *ignore = registration->cluster_resource_variable_ops_unsafely;
    }
    return absl::OkStatus();
}

absl::StatusOr<bool> MarkForCompilationPassImpl::Initialize()
{
    TF_RET_CHECK(!initialized_ && !edgesContracted_ && !clustersCreated_);
    initialized_ = true;

    TF_RETURN_IF_ERROR(FindCompilationCandidates());

    if (compilation_candidates_.empty()) {
        VLOG(VLOG_LEVEL_2) << "No compilable candidates";
        return false;
    }

    TF_ASSIGN_OR_RETURN(bool cycle_detection_graph_ok, CreateCycleDetectionGraph(graph_, &cycles_graph_));
    if (!cycle_detection_graph_ok) {
        VLOG(VLOG_LEVEL_2) << "Could not form cycle detection graph";
        return false;
    }

    if (!debug_options_.ignoreDeadnessChecks) {
        XLA_SCOPED_LOGGING_TIMER_LEVEL("DeadnessAnalysis", 1);
        TF_RETURN_IF_ERROR(DeadnessAnalysis::Run(*graph_, &deadness_analysis_));
    }

    // Each compilation candidate belongs to a cluster. The cluster's
    // representative names the node in the 'cycles' graph that represents the
    // cluster.
    TF_RETURN_IF_ERROR(BuildInitialClusterSet());
    return true;
}

template <typename FnTy>
absl::StatusOr<bool> MarkForCompilationPassImpl::ForEachEdgeInPostOrder(FnTy fn)
{
    bool changed = false;
    for (int32_t node : cycles_graph_.AllNodesInPostOrder()) {
        Cluster* cluster_from = GetClusterForCyclesGraphNode(node);
        if (!cluster_from) {
            continue;
        }

        // Make a copy of the set of successors because we may modify the graph in
        // TryToContractEdge.
        std::vector<int32> successors_copy = cycles_graph_.SuccessorsCopy(cluster_from->CyclesGraphNodeId());

        for (int to : successors_copy) {
            iterationCount_++;

            Cluster* cluster_to = GetClusterForCyclesGraphNode(to);
            if (!cluster_to) {
                continue;
            }

            TF_ASSIGN_OR_RETURN(bool contracted_edge, fn(cluster_from, cluster_to));
            changed |= contracted_edge;
        }
    }

    return changed;
}

Node* MarkForCompilationPassImpl::GetOnlyNodeIn(const Cluster& cluster)
{
    return cluster.ClusterSize() == 1 ? graph_->FindNodeId(cluster.GetIdOfOnlyNode()) : nullptr;
}

bool MarkForCompilationPassImpl::IsSinkLike(const Cluster& cluster)
{
    if (Node* n = GetOnlyNodeIn(cluster)) {
        return n->type_string() == "NoOp" && n->out_edges().size() == 1 && (*n->out_edges().begin())->dst()->IsSink();
    }

    return false;
}

bool MarkForCompilationPassImpl::IsScalarIntegerResourceOperation(const Cluster& cluster)
{
    Node* n = GetOnlyNodeIn(cluster);
    if (!n) {
        return false;
    }

    if (n->type_string() != "AssignAddVariableOp" && n->type_string() != "AssignSubVariableOp") {
        return false;
    }

    DataType dtype;
    if (!TryGetNodeAttr(n->def(), "dtype", &dtype) || !DataTypeIsInteger(dtype)) {
        return false;
    }

    Node* const_input = nullptr;
    for (const Edge* e : n->in_edges()) {
        if (!e->IsControlEdge() && e->src()->IsConstant()) {
            const_input = e->src();
            break;
        }
    }

    if (!const_input) {
        return false;
    }

    const TensorProto* proto = nullptr;
    if (!TryGetNodeAttr(const_input->def(), "value", &proto)) {
        return false;
    }

    return TensorShapeUtils::IsScalar(proto->tensor_shape());
}

Status MarkForCompilationPassImpl::RunEdgeContractionLoop()
{
    TF_RET_CHECK(initialized_ && !edgesContracted_ && !clustersCreated_);
    edgesContracted_ = true;

    // In general there are multiple maximal clusterings, but they are not all
    // equally performant.  Some clustering decision are likely to improve
    // performance much more than others, and we cannot order contractions on this
    // cost function, nor can we look at global information while deciding on
    // individual edges to contract.  Instead, we will make decisions on these
    // important edges then make decisions on all other edges, causing the highest
    // chance of all most important edges to be contracted.
    //
    // An example of where this might occur is with a digraph:
    // {A -> B, B -> C, A -> X, X -> C} where B is a Size operation and X is
    // not-compilable. In this case, the valid clusterings are {A,B} or {B,C}. B
    // should be clustered with A because it will prevent a potentially large
    // tensor from A being computed and copied.
    //
    // To choose better maximal clusterings we make multiple iterations over the
    // graph in post-order, where each such iteration is called a "phase".

    // Phase 0: contract metadata operations with their producer.

    VLOG(VLOG_LEVEL_4) << "Running phase 0";
    TF_RETURN_IF_ERROR(ForEachEdgeInPostOrder([&](Cluster* from, Cluster* to) -> absl::StatusOr<bool> {
                           // Shape consuming operations are desirable to cluster with their
                           // operands because they return a small set of scalar values after
                           // consuming a large amount of data.  For example, given a graph X -> Y
                           // -> Size -> Z, where the possible clustering is [{X, Y, Size}, {Z}] or
                           // [{X, Y}, {Size, Z}], the better clustering is Size with Y because the
                           // output of size will be a small tensor while Y is a potentially large
                           // tensor that must be computed and possible transposed/copied before
                           // the second cluster executes.
                           Node* n = GetOnlyNodeIn(*to);
                           bool is_shape_consumer_op = n && IsShapeConsumerOp(*n);
                           if (!is_shape_consumer_op) {
                               return false;
                           }

                           return TryToContractEdge(from, to);
                       }).status());

    // Phase 1: apply a heuristic to ensure that we don't mess up clustering due
    // to "group_deps".  After this phase most edges should have been contracted.

    VLOG(VLOG_LEVEL_4) << "Running phase 1";
    TF_RETURN_IF_ERROR(ForEachEdgeInPostOrder([&](Cluster* from, Cluster* to) -> absl::StatusOr<bool> {
                           // We split out this phase to get good clustering in the presence of a
                           // specific pattern seen in some graphs:
                           //
                           // digraph {
                           //   ApplyWeightUpdates_0 -> "iteration++"
                           //   ApplyWeightUpdates_1 -> "iteration++"
                           //   ApplyWeightUpdates_2 -> "iteration++"
                           //   ApplyWeightUpdates_0 -> Computation_A
                           //   ApplyWeightUpdates_1 -> Computation_B
                           //   ApplyWeightUpdates_2 -> Computation_C
                           //   Computation_A -> NoOp
                           //   Computation_B -> NoOp
                           //   Computation_C -> NoOp
                           //   "iteration++" -> NoOp
                           // }
                           //
                           // In the graph above we can't cluster iteration++ with any of the
                           // gradient update operations since that will break the TF resource
                           // variable memory model.  Given that constraint the ideal clustering
                           // would be to put all the gradient updates and all of the Computation_*
                           // nodes in one cluster, and leave iteration++ and NoOp unclustered.
                           //
                           // A naive post-order traversal would not create this good clustering,
                           // however.  Instead it will first create a cluster that puts
                           // Computation_* nodes, the NoOp and iteration++ node in a single
                           // cluster, after which it will fail to put any of the
                           // ApplyWeightUpdates_* nodes into this cluster. To avoid this fate we
                           // instead run a pass that avoids contracting edges _into_ NoOps like
                           // the above, and avoid clustering edges _from_ "iteration++" like the
                           // above.  Then we run a second pass that contracts the edges we could
                           // not contract the first time around.

                           if (IsSinkLike(*to)) {
                               return false;
                           }

                           if (IsScalarIntegerResourceOperation(*from)) {
                               return false;
                           }

                           return TryToContractEdge(from, to);
                       }).status());

    // Phase 2: contract any remaining edges.  After this phase we should have a
    // maximal clustering:
    //
    // A. We visit a cluster only after maximally clustering all its children.
    // B. By the time we're done with a node all of its children that could have
    //    been absorbed into the node have been absorbed.
    // C. We have an invariant that making a cluster larger does not make edges
    //    leaving it more contractable. That is, if we have
    //    digraph { X->Y; Y->Z; } then collapsing X->Y does not make it possible
    //    to contract Y->Z if Y->Z was not contractible originally.
    VLOG(VLOG_LEVEL_4) << "Running phase 2";
    TF_RETURN_IF_ERROR(
        ForEachEdgeInPostOrder([&](Cluster* from, Cluster* to) { return TryToContractEdge(from, to); }).status());

    // Check that the conclusion made above (that iterating over the graph once in
    // post order gives a maximal clustering) holds.  Once the linear time
    // post-order scheme has been battle tested we can move this to happen only in
    // debug builds.
    VLOG(VLOG_LEVEL_2) << "Checking idempotence";
    TF_ASSIGN_OR_RETURN(
        bool changed, ForEachEdgeInPostOrder([&](Cluster* from, Cluster* to) { return TryToContractEdge(from, to); }));
    TF_RET_CHECK(!changed);

    return absl::OkStatus();
}

std::atomic<int64_t> g_clusterSequenceNum;

int64_t GetNextClusterSequenceNumber()
{
    return g_clusterSequenceNum++;
}

Status MarkForCompilationPassImpl::CreateClusters()
{
    TF_RET_CHECK(initialized_ && edgesContracted_ && !clustersCreated_);
    clustersCreated_ = true;

    // Names for each cluster.
    std::unordered_map<int, string> cluster_names;

    if (debug_options_.dumpGraphs) {
        dump_graph::DumpGraphToFile("before_mark_for_compilation", *graph_, flib_def_);
    }

    // Mark clusters for compilation that:
    // * are placed on a device that requires compilation (an XlaDevice),
    // * are explicitly marked for compilation (_XlaCompile=true), or
    // * have more than debug_options_.xla_minClusterSize elements (applicable
    //   only if compilation is enabled, otherwise there will be no such
    //   candidates).
    for (Node* n : compilation_candidates_) {
        Cluster* cluster = GetClusterForNode(n);
        TF_ASSIGN_OR_RETURN(bool should_compile_cluster, ShouldCompileCluster(*cluster));
        if (!should_compile_cluster) {
            continue;
        }

        // We assume that functional If and While nodes have at least
        // minClusterSize non-trivial nodes in them.  It would be more principled
        // to (recursively) verify this fact, but that's probably not worth the
        // trouble.

        if (cluster->EffectiveClusterSize() >= debug_options_.minClusterSize || cluster->HasFunctionalControlFlow() ||
            cluster->IsXlaCompileAttrTrue()) {
            string& name = cluster_names[cluster->CyclesGraphNodeId()];

            if (name.empty()) {
                name = absl::StrCat("npu_xla_cluster_", GetNextClusterSequenceNumber());
            }

            n->AddAttr(kXlaClusterAttr, name);
            n->AddAttr(K_XLA_ALREADY_CLUSTERED, true);
            VLOG(VLOG_LEVEL_3) << "Assigning node " << n->name() << " to cluster " << name;
        } else {
            VLOG(VLOG_LEVEL_2) << "Rejecting node for small cluster: " << n->name()
                               << ", cluster size: " << cluster->EffectiveClusterSize()
                               << ", minClusterSize: " << debug_options_.minClusterSize
                               << ", cluster: " << cluster->CyclesGraphNodeId();
        }
    }

    return absl::OkStatus();
}

Status MarkForCompilationPassImpl::DumpDebugInfo()
{
    TF_RET_CHECK(initialized_ && edgesContracted_ && clustersCreated_);

    if (VLOG_IS_ON(VLOG_LEVEL_2)) {
        DumpPostClusteringGraphs();
    }

    VLogClusteringSummary();

    return absl::OkStatus();
}

}  // namespace npu_xla
}  // namespace tensorflow
