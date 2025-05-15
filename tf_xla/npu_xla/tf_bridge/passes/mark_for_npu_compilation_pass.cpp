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

absl::StatusOr<bool> MarkForCompilationPassImpl::ClusteringWillIntroduceInterDeviceDependency(
    const Cluster& cluster_from, const Cluster& cluster_to)
{
    // If any of the consumer's producers are on a different device, do not
    // cluster these nodes. This prevents other work on this device from being
    // delayed by work on other devices. We consider predecessors of the entire
    // cluster rather than just the inputs to the node to prevent the cluster
    // still being combined in cases where the 'to' cluster has multiple
    // dependencies on the 'from' cluster and another dependency leads to a
    // merging of the clusters.
    for (const auto& in_id : cycles_graph_.Predecessors(cluster_to.CyclesGraphNodeId())) {
        const Cluster* cluster_in = GetClusterForCyclesGraphNode(in_id);
        if (cluster_in) {
            TF_ASSIGN_OR_RETURN(bool devices_compatible, AreDevicesCompatible(cluster_to, *cluster_in));
            if (!devices_compatible) {
                return true;
            }
            TF_ASSIGN_OR_RETURN(devices_compatible, AreDevicesCompatible(cluster_from, *cluster_in));
            if (!devices_compatible) {
                return true;
            }
        }
    }

    return false;
}

std::optional<string> MarkForCompilationPassImpl::GetXlaScope(Node* node)
{
    // Look for either _XlaScope or _XlaInternalScope on both nodes to guide
    // clustering.  If both nodes have a scope and the scopes do not match, do
    // not cluster along this edge.  If even one of the nodes lacks a scope
    // attribute, then it is treated as a "bridge" and a cluster may be created
    // along it.
    //
    // The difference between _XlaScope and _XlaInternalScope is that _XlaScope is
    // provided by users through jit_scope APIs, while _XlaInternalScope is
    // automatically generated by the ClusterScopingPass when auto_jit is on.  As
    // such, we respect _XlaScope only when auto_jit is off, while respecting
    // _XlaInternalScope only when auto_jit is on.
    //
    // We may want to restrict the _XlaScope behavior to require all nodes marked
    // with _XlaCompile=true to also have a _XlaScope property set (and raise an
    // error otherwise); but for now we don't do this.

    if (global_jit_level_ != OptimizerOptions::OFF) {
        // If global_jit_level_ is ON, respect only _XlaInternalScope.
        const string& scope = GetNodeAttrString(node->attrs(), kXlaInternalScopeAttr);
        if (!scope.empty()) {
            return scope;
        }
    } else {
        // If global_jit_level_ is OFF, respect only _XlaScope.
        const string& scope = GetNodeAttrString(node->attrs(), kReuseXlaScopeAttr);
        if (!scope.empty()) {
            return scope;
        }
    }

    return std::nullopt;
}

// Returns true iff the attribute `attrName` is attached to either the node or
// to it's callee.
static bool GetNodeOrFuncAttr(Node* node, FunctionLibraryDefinition* flib_def, const char* attrName)
{
    bool out = false;
    bool attrValue;
    if (TryGetNodeAttr(node->attrs(), attrName, &attrValue)) {
        out |= attrValue;
    }

    if (flib_def->GetAttr(*node, attrName, &attrValue).ok()) {
        out |= attrValue;
    }
    return out;
}

Status MarkForCompilationPassImpl::BuildInitialClusterSet()
{
    auto ignoreResourceOps = [&device_info_cache_](const Node& n, bool* ignore) {
        return IgnoreResourceOpForSafetyAnalysis(&device_info_cache_, n, ignore);
    };

    std::vector<std::pair<int, int>> unsafe_resource_deps_vect;
    TF_RETURN_IF_ERROR(
        ComputeIncompatibleResourceOperationPairs(*graph_, flib_def_, ignoreResourceOps, &unsafe_resource_deps_vect));
    absl::c_copy(unsafe_resource_deps_vect, std::inserter(unsafe_resource_deps_, unsafe_resource_deps_.begin()));

    cluster_for_node_.resize(graph_->num_node_ids());
    for (Node* node : graph_->nodes()) {
        if (!IsCompilationCandidate(node)) {
            cluster_for_node_[node->id()].Get() = nullptr;
            continue;
        }

        // We want clusters to be big enough that the benefit from XLA's
        // optimizations offsets XLA related overhead (for instance we add some
        // Switch/Merge nodes into the graph to implement lazy compilation).  To
        // this end, we don't count Identity and Constant nodes because they do not
        // enable interesting optimizations by themselves.
        int effective_cluster_size = (node->IsIdentity() || node->IsConstant()) ? 0 : 1;

        bool has_functional_control_flow = node->IsWhileNode() || node->IsIfNode();

        std::optional<DeadnessPredicate> deadness_predicate;
        if (deadness_analysis_) {
            TF_ASSIGN_OR_RETURN(deadness_predicate, deadness_analysis_->GetPredicateFor(node, Graph::kControlSlot));
        }

        const string& device_name_str =
            !node->assigned_device_name().empty() ? node->assigned_device_name() : node->requested_device();
        TF_ASSIGN_OR_RETURN(DeviceId device, device_info_cache_.GetIdFor(device_name_str));

        bool is_resource_op = HasResourceInputOrOutput(*node);
        std::optional<DeviceId> resource_op_device;
        if (is_resource_op) {
            resource_op_device = device;
        }

        std::optional<int> resource_var_operation_node_id;
        if (is_resource_op || MayCallFunction(*node, flib_def_)) {
            resource_var_operation_node_id = node->id();
        }

        bool is_xla_compile_attr_true = GetNodeOrFuncAttr(node, flib_def_, kXlaCompileAttr) ||
                                        GetNodeOrFuncAttr(node, flib_def_, kXlaMustCompileAttr) ||
                                        GetNodeOrFuncAttr(node, flib_def_, kReuseXlaCompileAttr);

        DeviceSet devices;
        devices.Insert(device);

        Cluster* new_cluster =
            MakeNewCluster(node->id() /* cycles_graph_node_id= */, effective_cluster_size /* effective_cluster_size= */,
                           has_functional_control_flow /* has_functional_control_flow= */, devices, resource_op_device,
                           resource_var_operation_node_id, deadness_predicate,
                           is_xla_compile_attr_true /* is_xla_compile_attr_true= */, GetXlaScope(node));

        cluster_for_node_[node->id()].Get() = new_cluster;
    }

    return absl::OkStatus();
}

absl::StatusOr<bool> IsIdentityDrivingConstsInLoop(Node* node)
{
    if (!node->IsIdentity()) {
        return false;
    }

    // Check if the Identity is driven by a Switch on its true path.
    auto it =
        absl::c_find_if(node->in_edges(), [](const Edge* e) { return e->src()->IsSwitch() && e->src_output() == 1; });
    if (it == node->in_edges().end()) {
        return false;
    }
    const Node* switch_node = (*it)->src();

    // Check if the Switch is driven by LoopCond.
    const Node* maybe_loop_cond;
    TF_RETURN_IF_ERROR(switch_node->input_node(1, &maybe_loop_cond));
    if (!maybe_loop_cond->IsLoopCond()) {
        return false;
    }

    // Check if the Identity is driving any const nodes through a control edge.
    bool drivingAnyConsts =
        absl::c_any_of(node->out_edges(), [](const Edge* e) { return e->dst()->IsConstant() && e->IsControlEdge(); });
    if (!drivingAnyConsts) {
        return false;
    }

    return true;
}

std::unordered_set<string> GetOrCreateWhitelist()
{
    std::unordered_map<string, std::vector<string>>* whitelist_table = GetWhitelistTable();
    MarkForCompilationPassFlags* flags = GetMarkForCompilationPassFlags();
    std::unordered_set<string> whitelist;

    for (auto s : absl::StrSplit(flags->tf_xla_ops_to_cluster, ',')) {
        if (s == "FUSIBLE") {
            for (auto pair : *whitelist_table) {
                whitelist.insert(pair.second.begin(), pair.second.end());
            }
        } else {
            auto it = whitelist_table->find(std::string(s));
            if (it != whitelist_table->end()) {
                whitelist.insert(it->second.begin(), it->second.end());
            } else if (!s.empty()) {
                // Should be a user provided TF operation.
                whitelist.insert(string(s));
            }
        }
    }

    if (VLOG_IS_ON(VLOG_LEVEL_2) && !whitelist.empty()) {
        std::vector<string> vwhitelist(whitelist.begin(), whitelist.end());
        std::sort(vwhitelist.begin(), vwhitelist.end());
        VLOG(VLOG_LEVEL_2) << "XLA clustering will only consider the following TF operations: "
                           << absl::StrJoin(vwhitelist, " ");
    }
    return whitelist;
}

Status MarkForCompilationPassImpl::FindCompilationCandidates()
{
    OptimizerOptions opts;
    std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(
        new ProcessFunctionLibraryRuntime(nullptr, env_, nullptr /* config= */, TF_GRAPH_DEF_VERSION, flib_def_, opts));
    FunctionLibraryRuntime* lib_runtime = pflr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);
    std::vector<bool> compile_time_const_nodes(graph_->num_node_ids(), false);
    std::vector<bool> compile_time_fixed_shape_nodes(graph_->num_node_ids(), false);
    TF_RETURN_IF_ERROR(BackwardsConstAnalysis(
        *graph_, nullptr /* compile_time_const_arg_indices= */, &compile_time_const_nodes,
        nullptr /* compile_time_fixed_shape_arg_indices */,
        &compile_time_fixed_shape_nodes /* compile_time_fixed_shape_nodes */, lib_runtime,
        [](const Edge& e) { return true; }, debug_options_.clusterForMlir));
    // Iterate over nodes in sorted order so that compiler fuel is deterministic.
    // We can't simply pass op_nodes().begin() and op_nodes().end() to the
    // std::vector constructor because they're not proper iterators, with
    // iterator_traits defined and so on.
    std::vector<Node*> sorted_nodes;
    for (Node* node : graph_->op_nodes()) {
        sorted_nodes.push_back(node);
    }
    std::sort(sorted_nodes.begin(), sorted_nodes.end(), NodeComparatorID());

    if (*debug_options_.fuel >= std::numeric_limits<int64_t>::max() / 2) {
        // The assumption is that if fuel started out as INT64_MAX, it will forever
        // stay greater than INT64_MAX / 2.
        VLOG(VLOG_LEVEL_2) << "Starting fuel: infinity";
    } else {
        VLOG(VLOG_LEVEL_2) << "Starting fuel: " << *debug_options_.fuel;
    }

    VLOG(VLOG_LEVEL_2) << "sorted_nodes.size() = " << sorted_nodes.size();

    auto whitelist = GetOrCreateWhitelist();
    std::vector<std::string> vall_ops = XlaOpRegistry::GetAllRegisteredOps();
    VLOG(VLOG_LEVEL_2) << "registered xla ops: " << absl::StrJoin(vall_ops, ",");
    std::unordered_set<std::string> all_ops(vall_ops.begin(), vall_ops.end());
    // Check that user's provided TF operation really exists.
    for (const auto& s : whitelist) {
        if (!all_ops.count(string(s))) {
            return errors::InvalidArgument("The operation '", s,
                                           "' passed to --tf_xla_ops_to_cluster is not supported by NPU_XLA.");
        }
    }

    for (Node* node : sorted_nodes) {
        if (*debug_options_.fuel <= 0) {
            VLOG(1) << "Hit fuel limit; not marking any remaining ops as clusterable.";
            break;
        }

        TF_ASSIGN_OR_RETURN(const DeviceType& device_type,
                            device_info_cache_.GetDeviceTypeFor(node->assigned_device_name()));
        VLOG(VLOG_LEVEL_2) << "Device type for " << node->name() << ": " << device_type.type_string();

        if (CompilationDisallowedByXlaCompileAttr(node)) {
            VLOG(VLOG_LEVEL_2) << "Not clustering " << node->name() << ": disallowed by _XlaCompile attribute";
            continue;
        }

        const XlaOpRegistry::DeviceRegistration* registration;
        if (!XlaOpRegistry::GetCompilationDevice(device_type.type(), &registration)) {
            VLOG(VLOG_LEVEL_2) << "Rejecting " << node->name() << ": could not find JIT device for "
                               << device_type.type();
            continue;
        }

        DeviceType jit_device_type(registration->compilation_device_name);

        RecursiveCompilabilityChecker::OperationFilter op_filter = CreateOperationFilter(*registration);

        if (debug_options_.forceAllowTensorArrayOps) {
            op_filter.allow_tensor_array_ops = true;
            op_filter.allow_resource_ops_in_called_functions = true;
        }
        if (debug_options_.clusterForMlir) {
            op_filter.allow_stateful_rng_ops = true;
        }
        op_filter.skipClusteredOps = debug_options_.skipClusteredOps;

        if (!RecursiveCompilabilityChecker{&op_filter, &jit_device_type}.IsCompilableNode(*node, lib_runtime)) {
            VLOG(VLOG_LEVEL_2) << "Rejecting TF operation " << node->def().op() << " as it is not compilable";
            continue;
        }

        if (!whitelist.empty() && !whitelist.count(node->def().op())) {
            VLOG(VLOG_LEVEL_2) << "Rejecting TF operation " << node->def().op()
                               << " as it is not listed in --tf_xla_ops_to_cluster.";
            continue;
        }

        if (compile_time_const_nodes[node->id()]) {
            const OpDef* op_def;
            TF_RETURN_IF_ERROR(graph_->op_registry()->LookUpOpDef(node->type_string(), &op_def));
            if (op_def->is_stateful()) {
                // It is easiest to demonstrate the problem we're trying to solve with
                // an example.  Say we have this graph:
                //
                //   shape = RandomUniformInt();
                //   reshape = Reshape(input, shape)
                //
                // Both RandomUniformInt and Reshape are compilable by XLA so, absent
                // any other reason, we will try to put both shape and reshape in the
                // same cluster.  However, since XLA only supports statically shaped
                // values, it will expect to be able to constant fold `shape` to get a
                // static shape for `reshape`.  This is a problem because side-effecting
                // ops like RandomUniformInt() cannot be constant folded.  We fix this
                // by putting `shape` and `reshape` in different clusters, which results
                // in us recompiling `reshape`'s cluster for every new value of `shape`,
                // making `reshape` statically sized within each compilation.  We
                // simplify the solution even further by disallowing operations like
                // `shape` from being part of *any* non-trivial cluster.  They're either
                // not compiled by XLA altogether or, if assigned to an XLA_* device
                // with "must compile" semantics, compiled into a trivial single-op
                // cluster.  This approach leaves some room for improvement, and we can
                // consider implementing a more aggressive data-flow-analysis based
                // solution in the future if needed.
                //
                // One ugly problem we have to contend with: certain sets of ops *have*
                // to be in the same cluster because values flowing between them have
                // types that can't be live-in or live-out of a cluster.  These ops are:
                //
                //  - TensorArray ops operating on the same TensorArray instance.
                //  - Stack ops operating on the same Stack instance.
                //
                // To work around this we avoid isolating these specific ops.  Because
                // of this concession it is unsound to auto-cluster them because then
                // we'd create clusters we could not compile (because we can't constant
                // fold, say, a TensorArrayRead or a StackPopV2).  But we don't
                // auto-cluster these operations today so we're good for now.
                const XlaResourceOpInfo* op_info = GetResourceOpInfoForOp(node->type_string());
                bool is_tensor_array_or_stack_op = op_info && op_info->resource_kind() != XlaResourceKind::kVariable;
                if (!is_tensor_array_or_stack_op) {
                    VLOG(VLOG_LEVEL_2) << "Isolating " << node->name() << ": must-be-constant stateful op";
                    continue;
                }
            }
        }

        // This is a heuristic to avoid creating dependency between while loop
        // condition and body computations.  Dependency between them can be created
        // if a special Identity node in the following pattern is clustered in.
        // That is, an Identity node in the loop cond computation is used to drive
        // const nodes consumed by the loop body.  If this Identity node goes into
        // the same cluster with nodes from the loop body, extra dependency is
        // created between the loop cond and body computations and it hinders the
        // progression of the loop cond computation at runtime with significant
        // overhead.  Specifically, we look for the below pattern and do not cluster
        // in this Identity to avoid the described issue.  Since Identity has low
        // execution cost in native TF, the fact that this heuristic gives up these
        // special Identity nodes as candidates should not harm any performance.  If
        // other considerations emerge in the future, we can revisit the heuristic
        // and only disallow these Identities to go into the cluster with nodes from
        // the loop body but still consider them candidates.
        //
        // LoopCond ->
        // Merge    -> Switch -> Identity -> i++ -> ... -> NextIteration
        //                               ..> Const -> LoopBody
        //                            (control edge)
        TF_ASSIGN_OR_RETURN(bool is_identity_driving_consts_in_loop, IsIdentityDrivingConstsInLoop(node));
        if (is_identity_driving_consts_in_loop) {
            VLOG(VLOG_LEVEL_2) << "Rejecting " << node->name()
                               << ": including it can create dependencies between while loop "
                                  "condition and body computations with runtime overhead.";
            continue;
        }

        compilation_candidates_.insert(node);
        --(*debug_options_.fuel);
    }

    VLOG(VLOG_LEVEL_2) << "compilation_candidates_.size() = " << compilation_candidates_.size();
    return absl::OkStatus();
}

bool MarkForCompilationPassImpl::CompilationDisallowedByXlaCompileAttr(Node* node)
{
    if (debug_options_.ignoreXlaCompileAttr) {
        return false;
    }

    // If there is a _XlaCompile annotation, use its value.
    bool compile = false;
    Status status = GetNodeAttr(node->attrs(), kXlaCompileAttr, &compile);
    if (status.ok()) {
        if (!compile) {
            VLOG(VLOG_LEVEL_2) << "Rejecting " << node->name() << ": kXlaCompileAttr(" << kXlaCompileAttr
                               << ") is false.";
        }
        return !compile;
    }

    status = flib_def_->GetAttr(*node, kXlaCompileAttr, &compile);
    if (status.ok()) {
        if (!compile) {
            VLOG(VLOG_LEVEL_2) << "Rejecting " << node->name() << ": kXlaCompileAttr(" << kXlaCompileAttr
                               << ") on callee is false.";
        }
        return !compile;
    }

    return false;
}

bool MarkForCompilationPassImpl::LogNotContractableAndReturnFalse(Cluster* from, Cluster* to, absl::string_view reason)
{
    VLOG(VLOG_LEVEL_2) << EdgeContractionFailureMsg(from, to, reason);
    return false;
}

absl::StatusOr<bool> MarkForCompilationPassImpl::TryToContractEdge(Cluster* from, Cluster* to)
{
    DCHECK(from->deadness_predicate().has_value() == to->deadness_predicate().has_value());
    if (from->deadness_predicate() != to->deadness_predicate()) {
        VLOG(VLOG_LEVEL_2) << EdgeContractionFailureMsg(
            from, to,
            absl::StrCat("the two nodes have mismatching deadness: ",
                         deadness_analysis_->DebugString(*from->deadness_predicate()), " and ",
                         deadness_analysis_->DebugString(*to->deadness_predicate())));
        return false;
    }

    TF_ASSIGN_OR_RETURN(bool devices_compatible, AreDevicesCompatible(*from, *to));
    if (!devices_compatible) {
        return LogNotContractableAndReturnFalse(from, to, "the two nodes have incompatible devices");
    }

    if (from->xla_scope().has_value() && to->xla_scope().has_value() && *from->xla_scope() != *to->xla_scope()) {
        return LogNotContractableAndReturnFalse(from, to, "the two nodes have mismatching XLA scopes");
    }

    // Don't exceed the maximum cluster size.
    if (from->ClusterSize() + to->ClusterSize() > debug_options_.maxClusterSize) {
        return LogNotContractableAndReturnFalse(from, to, "the new cluster will be larger than the max cluster size");
    }

    TF_ASSIGN_OR_RETURN(bool will_introduce_cross_device_dependency,
                        ClusteringWillIntroduceInterDeviceDependency(*from, *to));

    if (will_introduce_cross_device_dependency) {
        return LogNotContractableAndReturnFalse(from, to, "the new cluster will introduce a cross device dependency");
    }

    // Check if contracting this edge will break the resource variable concurrency
    // semantics.  In theory this is quadratic in the number of nodes, but seems
    // to not be a problem in practice so far.
    if (!debug_options_.ignoreResourceVariableChecks) {
        for (int resource_var_from : from->resource_var_operation_node_ids()) {
            for (int resource_var_to : to->resource_var_operation_node_ids()) {
                // If unsafe_resource_deps_ contains {A, B} then
                //
                //  a. A and B are resource operations.
                //  b. A and B cannot be placed in the same cluster.
                //  c. There is no path from B to A in the cycles graph (but there may
                //     be a path from A to B).
                //
                // So check the legality of the edge contraction by checking if any of
                // the n^2 pairs of resource variable operations are forbidden.
                if (unsafe_resource_deps_.contains({resource_var_from, resource_var_to})) {
                    return LogNotContractableAndReturnFalse(from, to,
                                                            "the new cluster would break resource variable semantics");
                }
            }
        }
    }

    return MergeClusters(from, to);
}

Status MarkForCompilationPassImpl::Run()
{
    // Make sure that kernels have been registered on the JIT device.
    XlaOpRegistry::RegisterCompilationKernels();

    // Start the timer after XlaOpRegistry::RegisterCompilationKernels which does
    // some one-time work.
    XLA_SCOPED_LOGGING_TIMER_LEVEL("MarkForCompilationPassImpl::Run", 1);

    TF_ASSIGN_OR_RETURN(bool initialized, Initialize());
    if (!initialized) {
        // Initialization exited early which means this instance of
        // MarkForCompilationPassImpl is not set up to run the subsequent phases.
        return absl::OkStatus();
    }

    TF_RETURN_IF_ERROR(RunEdgeContractionLoop());
    TF_RETURN_IF_ERROR(CreateClusters());
    TF_RETURN_IF_ERROR(DumpDebugInfo());

    return absl::OkStatus();
}

void MarkForCompilationPassImpl::DumpPostClusteringGraphs()
{
    VLOG(VLOG_LEVEL_2) << dump_graph::DumpGraphToFile("mark_for_compilation", *graph_, flib_def_);

    // We also dump out an annotated version of the TF graph where the nodes
    // names are prefixed with the cluster names.  This can help visualizing the
    // clustering decisions on TensorBoard.
    Graph new_graph(graph_->op_registry());
    CopyGraph(*graph_, &new_graph);

    for (Node* n : new_graph.nodes()) {
        if (std::optional<absl::string_view> cluster_name = GetXlaClusterForNode(*n)) {
            n->set_name(absl::StrCat(*cluster_name, "/", n->name()));
        } else if (n->type_string() == "VarHandleOp") {
            n->set_name(absl::StrCat("varhandle/", n->name()));
        } else {
            // There is room for improvement here.  In particular, it may help to
            // split these unclustered nodes into classes where every node in a
            // specific class has edges to and from the same set of clusters.
            n->set_name(absl::StrCat("unclustered/", n->name()));
        }
    }

    VLOG(VLOG_LEVEL_2) << dump_graph::DumpGraphToFile("mark_for_compilation_annotated", new_graph, flib_def_);
}

string RatioToString(int numerator, int denominator)
{
    if (denominator == 0) {
        throw std::invalid_argument("denominator is 0");
    }
    return absl::StrFormat("%d / %d (%.2f%%)", numerator, denominator, (100.0 * numerator) / denominator);
}

void MarkForCompilationPassImpl::VLogClusteringSummary()
{
    if (!VLOG_IS_ON(VLOG_LEVEL_2)) {
        return;
    }

    XlaAutoClusteringSummary auto_clustering_info = GetXlaAutoClusteringSummary(*graph_);

    VLOG(VLOG_LEVEL_2) << "*** Clustering info for graph of size " << graph_->num_nodes();
    VLOG(VLOG_LEVEL_2) << " Built " << auto_clustering_info.clusters_size() << " clusters, size "
                       << RatioToString(auto_clustering_info.clustered_node_count(), graph_->num_nodes());

    for (const XlaAutoClusteringSummary::Cluster& cluster : auto_clustering_info.clusters()) {
        absl::string_view cluster_name = cluster.name();
        int size = cluster.size();
        VLOG(VLOG_LEVEL_2) << "  " << cluster_name << " " << RatioToString(size, graph_->num_nodes());
        for (const XlaAutoClusteringSummary::OpAndCount& op_count : cluster.op_histogram()) {
            VLOG(VLOG_LEVEL_3) << "   " << op_count.op() << ": " << op_count.count() << " instances";
        }
    }

    if (!auto_clustering_info.unclustered_op_histogram().empty()) {
        VLOG(VLOG_LEVEL_2) << " Unclustered nodes: "
                           << RatioToString(auto_clustering_info.unclustered_node_count(), graph_->num_nodes());
        for (const XlaAutoClusteringSummary::OpAndCount& op_count : auto_clustering_info.unclustered_op_histogram()) {
            VLOG(VLOG_LEVEL_3) << "  " << op_count.op() << ": " << op_count.count() << " instances";
        }
    }

    struct EdgeInfo {
        absl::string_view node_name;
        std::optional<absl::string_view> cluster_name;

        absl::string_view GetClusterName() const
        {
            return cluster_name ? *cluster_name : "[none]";
        }

        std::pair<absl::string_view, std::optional<absl::string_view>> AsPair() const
        {
            return {node_name, cluster_name};
        }

        bool operator<(const EdgeInfo& other) const
        {
            return AsPair() < other.AsPair();
        }
    };

    using EdgeInfoMap = std::map<absl::string_view, std::map<EdgeInfo, int64_t>>;

    EdgeInfoMap incoming_edge_infos;
    EdgeInfoMap outgoing_edge_infos;

    std::set<absl::string_view> cluster_names_to_print;

    for (const Edge* e : graph_->edges()) {
        const Node* from = e->src();
        std::optional<absl::string_view> from_cluster_name = GetXlaClusterForNode(*from);

        const Node* to = e->dst();
        std::optional<absl::string_view> to_cluster_name = GetXlaClusterForNode(*to);

        if (to_cluster_name == from_cluster_name) {
            continue;
        }

        if (to_cluster_name) {
            incoming_edge_infos[*to_cluster_name][EdgeInfo{from->name(), from_cluster_name}]++;
            cluster_names_to_print.insert(*to_cluster_name);
        }

        if (from_cluster_name) {
            outgoing_edge_infos[*from_cluster_name][{to->name(), to_cluster_name}]++;
            cluster_names_to_print.insert(*from_cluster_name);
        }
    }

    VLOG(VLOG_LEVEL_4) << "*** Inter-Cluster edges:";
    if (cluster_names_to_print.empty()) {
        VLOG(VLOG_LEVEL_4) << "   [none]";
    }

    auto printEdgeInfoSetForCluster = [](absl::string_view cluster_name, const EdgeInfoMap& edge_info_map,
                                         absl::string_view desc) {
        auto it = edge_info_map.find(cluster_name);
        if (it != edge_info_map.end()) {
            VLOG(VLOG_LEVEL_4) << "  " << it->second.size() << " " << desc << " edges";
            for (const auto& edge_info_count_pair : it->second) {
                VLOG(VLOG_LEVEL_4) << "   " << edge_info_count_pair.first.GetClusterName() << " "
                                   << edge_info_count_pair.first.node_name << " # " << edge_info_count_pair.second;
            }
        } else {
            VLOG(VLOG_LEVEL_4) << "  No " << desc << " edges.";
        }
    };

    for (absl::string_view cluster_name : cluster_names_to_print) {
        VLOG(VLOG_LEVEL_4) << " ** Cluster " << cluster_name;
        printEdgeInfoSetForCluster(cluster_name, incoming_edge_infos, "incoming");
        printEdgeInfoSetForCluster(cluster_name, outgoing_edge_infos, "outgoing");
    }
}

absl::StatusOr<bool> MarkForCompilationPassImpl::AreDevicesCompatible(const Cluster& cluster_a,
                                                                      const Cluster& cluster_b)
{
    DeviceSet devices = cluster_a.devices();
    devices.UnionWith(cluster_b.devices());

    TF_ASSIGN_OR_RETURN(std::optional<jit::DeviceId> maybe_chosen_device,
                        MaybePickDeviceForXla(device_info_cache_, devices, false /* allow_mixing_unknown_and_cpu= */));
    if (!maybe_chosen_device.has_value()) {
        return false;
    }

    jit::DeviceId chosen_device = *maybe_chosen_device;

    // If we are able to pick a device `chosen_device` for the larger cluster, the
    // resource operations in `cluster_a` and `cluster_b` must be placed on the
    // same device as `chosen_device`.  This is because the _XlaCompile and
    // _XlaRun kernels are going to run on and therefore try to access the
    // resource variables from `chosen_device`, which will be an error if the
    // resource variables are placed on some other device.
    auto resourceOpDeviceOk = [&chosen_device](std::optional<DeviceId> resource_op_device) {
        return !resource_op_device.has_value() || *resource_op_device == chosen_device;
    };

    return resourceOpDeviceOk(cluster_a.resource_op_device()) && resourceOpDeviceOk(cluster_b.resource_op_device());
}

// Returns `true` iff we should compile `cluster`.
absl::StatusOr<bool> MarkForCompilationPassImpl::ShouldCompileClusterImpl(const Cluster& cluster)
{
    TF_ASSIGN_OR_RETURN(DeviceId chosen_device, PickDeviceForXla(device_info_cache_, cluster.devices(),
                                                                 false /* allow_mixing_unknown_and_cpu= */));

    const DeviceType& device_type = device_info_cache_.GetDeviceTypeFor(chosen_device);
    const XlaOpRegistry::DeviceRegistration* registration = device_info_cache_.GetCompilationDevice(chosen_device);
    TF_RET_CHECK(registration) << "chosen device = " << device_info_cache_.GetNameFor(chosen_device)
                               << "; device type = " << device_type.type() << "; devices ("
                               << device_info_cache_.DebugString(cluster.devices());

    auto policy = registration->autoclustering_policy;
    bool shouldCompile =
        cluster.IsXlaCompileAttrTrue() ||
        registration->autoclustering_policy == XlaOpRegistry::AutoclusteringPolicy::kAlways ||
        (registration->autoclustering_policy == XlaOpRegistry::AutoclusteringPolicy::kIfEnabledGlobally &&
         global_jit_level_ != OptimizerOptions::OFF);
    if (!shouldCompile && global_jit_level_ != OptimizerOptions::OFF && device_type.type_string() == DEVICE_CPU) {
        static absl::once_flag once;
        absl::call_once(once, [] {
            LOG(WARNING) << "(One-time warning): Not using XLA:CPU for cluster because envvar "
                            "TF_XLA_FLAGS=--tf_xla_cpu_global_jit was not set.  If you want "
                            "XLA:CPU, either set that envvar, or use experimental_jit_scope "
                            "to enable XLA:CPU.  To confirm that XLA is active, pass "
                            "--vmodule=xla_compilation_cache=1 (as a proper command-line "
                            "flag, not via TF_XLA_FLAGS) or set the envvar "
                            "XLA_FLAGS=--xla_hlo_profile.";
            MarkForCompilationPassFlags* flags = GetMarkForCompilationPassFlags();
            if (flags->tf_xla_cpu_global_jit) {
                LOG(WARNING) << "(Although the tf_xla_cpu_global_jit flag is currently enabled, "
                                "perhaps it wasn't enabled at process startup?)";
            }
        });
    }

    VLOG(VLOG_LEVEL_2) << (shouldCompile ? "Compiling" : "Not compiling") << " cluster with device "
                       << device_info_cache_.GetNameFor(chosen_device);

    return shouldCompile;
}

absl::StatusOr<bool> MarkForCompilationPassImpl::ShouldCompileCluster(const Cluster& cluster)
{
    auto it = should_compile_cluster_cache_.find(&cluster);
    if (it != should_compile_cluster_cache_.end()) {
        return it->second;
    }

    TF_ASSIGN_OR_RETURN(bool should_compile, ShouldCompileClusterImpl(cluster));
    should_compile_cluster_cache_.insert({&cluster, should_compile});
    return should_compile;
}

bool IsSupportedResourceForXLA(Node* resource_op)
{
    if (resource_op == nullptr) {
        return false;
    }
    // Skip trivial ops (e.g. Identity)
    while (resource_op->type_string() == "Enter" || resource_op->type_string() == "Identity") {
        TF_CHECK_OK(resource_op->input_node(0, &resource_op));
    }
    const XlaResourceOpInfo* op_info = GetResourceOpInfoForOp(resource_op->type_string());
    return op_info && op_info->resource_kind() == XlaResourceKind::kVariable;
}

Status FilterInvalidClusters(Graph* graph, std::unordered_set<string>* invalid_clusters)
{
    VLOG(VLOG_LEVEL_2) << "FilterInvalidClusters...";

    CHECK(invalid_clusters != nullptr);
    invalid_clusters->clear();

    for (Node* node : graph->op_nodes()) {
        string name;
        if (!GetNodeAttr(node->attrs(), kXlaClusterAttr, &name).ok() || name.empty()) {
            // Skip if not in any XLA cluster
            continue;
        }

        // Detect cross-cluster non-variable resource edge
        auto is_cross_cluster_non_variable_resource_edge = [&name](const Edge* edge, Node* target, DataType dtype) {
            // Skip if not data resource edge
            if (edge->IsControlEdge() || dtype != DT_RESOURCE || IsSupportedResourceForXLA(target)) {
                return false;
            }
            // Skip if the edge is in the cluster.
            string target_name;
            if (GetNodeAttr(target->attrs(), kXlaClusterAttr, &target_name).ok() && target_name == name) {
                return false;
            }
            return true;
        };

        for (const Edge* edge : node->in_edges()) {
            if (is_cross_cluster_non_variable_resource_edge(edge, edge->src(), node->input_type(edge->dst_input()))) {
                invalid_clusters->insert(name);
            }
        }
        for (const Edge* edge : node->out_edges()) {
            if (is_cross_cluster_non_variable_resource_edge(edge, edge->dst(), node->output_type(edge->src_output()))) {
                invalid_clusters->insert(name);
            }
        }
    }

    for (auto& name : *invalid_clusters) {
        VLOG(VLOG_LEVEL_2) << "Find invalid cluster: " << name;
    }
    return absl::OkStatus();
}

// Clean all device information for nodes may have connection to `initial_nodes`
// through resource edge or colocate. This ensures Place can run successfully
// without conflict placement decision.
Status RecursivelySetDevice(const Graph* graph, const std::vector<Node*>& initial_nodes, const std::string& device_name,
                            bool consider_colocate)
{
    std::vector<Node*> unsafe_nodes(initial_nodes);
    std::vector<std::pair<Node*, Node*>> unsafe_edges;
    std::unordered_map<absl::string_view, Node*, hash<absl::string_view>> name_to_node;
    const absl::string_view kColocationAttrName("_class");
    const absl::string_view kColocationGroupPrefix("loc:@");

    auto isResourceEdge = [](const Edge* e) {
        if (e->IsControlEdge()) {
            return false;
        }
        auto dst = e->dst();
        DataType input_type = dst->input_type(e->dst_input());
        return (input_type == DT_RESOURCE || IsRefType(input_type));
    };
    auto parseColocateNodes = [&kColocationAttrName, &kColocationGroupPrefix, &name_to_node](
                                  const Node* node, std::vector<Node*>* colocate_nodes) {
        const AttrValue* attr_value = node->attrs().Find(kColocationAttrName);
        if (attr_value != nullptr && attr_value->has_list()) {
            for (const absl::string_view& class_spec : attr_value->list().s()) {
                absl::string_view spec(class_spec);
                if (absl::ConsumePrefix(&spec, kColocationGroupPrefix)) {
                    auto i = name_to_node.find(spec);
                    if (i != name_to_node.end()) {
                        colocate_nodes->push_back(i->second);
                    }
                }
            }
        }
    };
    auto isPartiallyMoved = [&unsafe_nodes](const Node* src, const Node* dst) {
        int nFound = std::count_if(unsafe_nodes.begin(), unsafe_nodes.end(),
                                   [=](const Node* node) { return node == src || node == dst; });
        return nFound == 1;
    };
    auto onSameDevice = [](const Node* src, const Node* dst) {
        string src_device = src->assigned_device_name();
        string dst_device = dst->assigned_device_name();
        return src_device == dst_device;
    };
    auto isClustered = [](const Node* node) {
        string cluster_name;
        return (GetNodeAttr(node->attrs(), kXlaClusterAttr, &cluster_name).ok() && !cluster_name.empty());
    };

    // Construct a map from string to Node as colocate is stored as string
    for (Node* node : graph->op_nodes()) {
        if (isClustered(node)) {
            continue;
        }
        name_to_node[node->name()] = node;
    }

    // Collect all "colocate edge". "Colocate edge" is made-up concepts. Placer
    // use a complex recursive algorithm to analyze colocate group. All nodes
    // within a colocate group have the same device. Porting this part is too
    // heavy. A easy way is to think colocate as edges, and we just make sure
    // src and dst node of an edge are on the same device. This make colocate
    // consistent with resource edge and easy to implement.
    if (consider_colocate) {
        for (auto it = name_to_node.begin(); it != name_to_node.end(); ++it) {
            Node* node = it->second;
            std::vector<Node*> colocate_nodes;
            parseColocateNodes(node, &colocate_nodes);
            for (Node* colocate_node : colocate_nodes) {
                unsafe_edges.push_back({node, colocate_node});
            }
        }
    }

    // Collect all resource edge
    for (Edge* edge : graph->edges()) {
        Node* src = edge->src();
        Node* dst = edge->dst();
        if (isClustered(src) || isClustered(dst)) {
            continue;
        }
        if (isResourceEdge(edge)) {
            unsafe_edges.push_back({src, dst});
        }
    }

    VLOG(VLOG_LEVEL_2) << "Total unsafe edges: " << unsafe_edges.size();

    int maxLoop = 1000000;
    int loopCnt = 0;
    while (true) {
        bool foundNewUnsafeNodes = false;

        for (auto& edge : unsafe_edges) {
            Node* src = edge.first;
            Node* dst = edge.second;
            if (isPartiallyMoved(src, dst)) {
                Node* node_to_move = (std::count(unsafe_nodes.begin(), unsafe_nodes.end(), dst) == 0) ? dst : src;
                unsafe_nodes.push_back(node_to_move);
                foundNewUnsafeNodes = true;
            }
        }

        if (!foundNewUnsafeNodes) {
            break;
        }

        ++loopCnt;
        if (loopCnt > maxLoop) {
            throw std::runtime_error("exceed max loop: " + std::to_string(maxLoop));
        }
    }

    VLOG(VLOG_LEVEL_2) << "Total unsafe nodes: " << unsafe_nodes.size();

    for (Node* node : unsafe_nodes) {
        if (std::count(initial_nodes.begin(), initial_nodes.end(), node) == 0) {
            // Clean all device information. Placer will help to figure this out.
            node->set_requested_device(device_name);
            node->set_assigned_device_name(device_name);
            VLOG(VLOG_LEVEL_2) << "Clean device information for " << node->name();
        }
    }
    return absl::OkStatus();
}

// This is used to update invalid nodes and clusters when enable control flow,
// which includes:
//   1. decluster invalid clusters
//   2. lower non-clustered control flow ops
//   3. move int32 TensorArray ops to CPU
// Return true if 2 and 3 happens.
StatusOr<bool> MaybeDeclusterAndLower(const GraphOptimizationPassOptions& options,
                                      const std::unordered_set<string>& invalid_clusters)
{
    Graph* graph = options.graph->get();

    bool isChanged = false;
    std::vector<Node*> nodes_to_move;

    // This ensures that nested If/While nodes and TensorArray ops can be handled.
    for (int i = 2; i < graph->num_node_ids(); ++i) {
        Node* node = graph->FindNodeId(i);
        if (node == nullptr)
            continue;  // deleted node

        string cluster_name;
        if (GetNodeAttr(node->attrs(), kXlaClusterAttr, &cluster_name).ok() && !cluster_name.empty()) {
            if (invalid_clusters.count(cluster_name) == 0) {
                continue;  // Skip nodes in valid clusters
            }
            node->ClearAttr(kXlaClusterAttr);
            node->ClearAttr(K_XLA_ALREADY_CLUSTERED);
            VLOG(VLOG_LEVEL_2) << node->name() << " is declustered";
        }

        if (IsFunctionalControlFlowOps(node)) {
            VLOG(VLOG_LEVEL_2) << node->name() << " is defunctionalized";
            TF_RETURN_IF_ERROR(DefunctionalizeFactory().defunctionalize(node, graph, *options.flib_def));
            isChanged = true;
        } else if (IsInvalidTensorArrayOps(node)) {
            nodes_to_move.push_back(node);
            isChanged = true;
        }
    }

    // Choose the first found CPU device. Note the original Placer also use this
    // policy to assign device, so it should not bring any performane penalty.
    string valid_host_device("");
    for (auto device : options.device_set->devices()) {
        DeviceType device_type("");
        TF_RETURN_IF_ERROR(DeviceNameToDeviceType(device->name(), &device_type));
        if (device_type == DEVICE_CPU) {
            valid_host_device = device->name();
            break;
        }
    }
    CHECK(valid_host_device != "");

    if (nodes_to_move.size() > 0) {
        // Force invalid TensorArray ops on CPU.
        for (Node* node : nodes_to_move) {
            node->set_requested_device("");
            node->set_assigned_device_name(valid_host_device);
            VLOG(VLOG_LEVEL_2) << "Place " << node->name() << " on " << valid_host_device;
        }

        TF_RETURN_IF_ERROR(
            RecursivelySetDevice(graph, nodes_to_move, valid_host_device, false /* consider_colocate */));
    }

    if (isChanged) {
        auto session_options = options.session_options;
        Placer placer(graph, "", options.flib_def, options.device_set, nullptr /* default_local_device= */,
                      session_options->config.allow_soft_placement(), session_options->config.log_device_placement());
        TF_CHECK_OK(placer.Run());
    }

    return isChanged;
}

Status MarkForCompilation(const GraphOptimizationPassOptions& options,
                          MarkForCompilationPassImpl::DebugOptions& debug_options)
{
    Graph* graph = options.graph->get();
    FunctionLibraryDefinition* flib_def = options.flib_def;

    // Deadness analysis expects a graph with source and sink edges properly
    // connected but sometimes the incoming graph does not follow this invariant.
    // So fix up the source and sink edges before calling into deadness analysis.
    FixupSourceAndSinkEdges(graph);

    for (Node* n : graph->nodes()) {
        // See explanation on `K_XLA_ALREADY_CLUSTERED`.
        if (n->attrs().Find(K_XLA_ALREADY_CLUSTERED)) {
            return absl::OkStatus();
        }
    }

    // By default, XLA does not auto-cluster resource ops (e.g. TensorArrays).
    // But XLA does support these ops if they are totally within an XLA cluster.
    // Functional while/if ops typically have tensorarray inputs, thus are
    // likely not auto-clustered by the default policy to avoid cross-cluster
    // non-vairable resource edge.
    //
    // To solve this problem, we first enable auto-cluster resource ops and ignore
    // resource variable checks. Then we will filter out invalid clusters. For
    // those invalid clusters, we further lower functional while/if ops to
    // traditional data control flow version and do another cluster pass without
    // resource ops auto-cluster support.
    //
    // When set tao_enable_control_flow, int32 TensorArray can be placed on GPU.
    // If there are not clustered, we need to move them back to CPU, as there is
    // no registered kernel for int32 TensorArray on GPU.
    bool enable_npu_xla = GetTfBridgeOptions()->enable_npu_xla;
    bool enable_control_flow = GetTfBridgeOptions()->enable_control_flow;
    if (enable_npu_xla && (HasFunctionalControlFlowOps(graph) || enable_control_flow)) {
        debug_options.forceAllowTensorArrayOps = true;
        debug_options.ignoreResourceVariableChecks = true;
        TF_RETURN_IF_ERROR(MarkForCompilationPassImpl{
            debug_options, graph, flib_def,
            options.session_options != nullptr ? options.session_options->env : Env::Default(),
            GetGlobalJitLevelForGraph(options)}
                               .Run());

        // XLA can not handle cross-cluster edge having non-variable resource
        // type. This can not be checked unless we know the clustering result.
        // Thus we do it here and revert those are not valid.
        std::unordered_set<string> invalid_clusters;
        TF_RETURN_IF_ERROR(FilterInvalidClusters(graph, &invalid_clusters));

        TF_ASSIGN_OR_RETURN(bool is_changed, MaybeDeclusterAndLower(options, invalid_clusters));

        if (is_changed) {
            debug_options.skipClusteredOps = true;
            // For safety, do not cluster any TensorArray in the second round
            debug_options.forceAllowTensorArrayOps = false;
            debug_options.ignoreResourceVariableChecks = false;
            return MarkForCompilationPassImpl{
                debug_options, graph, flib_def,
                options.session_options != nullptr ? options.session_options->env : Env::Default(),
                GetGlobalJitLevelForGraph(options)}
                .Run();
        } else {
            return absl::OkStatus();
        }
    } else {
        return MarkForCompilationPassImpl{
            debug_options, graph, flib_def,
            options.session_options != nullptr ? options.session_options->env : Env::Default(),
            GetGlobalJitLevelForGraph(options)}
            .Run();
    }
}

std::atomic<int64_t>* GetPointerToFuel(int64_t initial_value)
{
    static std::atomic<int64_t>* fuel = [&]() {
        std::atomic<int64_t>* fuel = new std::atomic<int64_t>;
        *fuel = initial_value;
        return fuel;
    }();

    return fuel;
}

}  // anonymous namespace

Status MarkForNpuCompilationPass::Run(const GraphOptimizationPassOptions& options)
{
    MarkForCompilationPassFlags* flags = GetMarkForCompilationPassFlags();

    MarkForCompilationPassImpl::DebugOptions debug_options;
    debug_options.ignoreDeadnessChecks = flags->tf_xla_disable_deadness_safety_checks_for_debugging;
    debug_options.ignoreResourceVariableChecks = flags->tf_xla_disable_resource_variable_safety_checks_for_debugging;
    debug_options.ignoreXlaCompileAttr = false;
    debug_options.maxClusterSize = flags->tf_xla_maxClusterSize;
    debug_options.minClusterSize = flags->tf_xla_minClusterSize;
    debug_options.fuel = GetPointerToFuel(flags->tf_xla_clustering_fuel);
    debug_options.dumpGraphs = flags->tf_xla_clustering_debug;

    return MarkForCompilation(options, debug_options);
}

std::vector<string> GetNpuXlaSupportedOps()
{
    std::vector<string> ops;
    // clang-format off
  ops.insert(ops.end(), {
    "Abs",
    "Add",
    "AddN",
    "All",
    "Any",
    "BatchMatMul",
    "BiasAdd",
    "BiasAddGrad",
    "BroadcastTo",
    "Cast",
    "Ceil",
    "ConcatV2",
    "Const",
    "Conv2D",
    "Cos",
    "DepthwiseConv2dNative",
    "DynamicStitch",
    "Equal",
    "Erf",
    "Exp",
    "ExpandDims",
    "Fill",
    "Floor",
    "FloorDiv",
    "FloorMod",
    "GatherNd",
    "Greater",
    "GreaterEqual",
    "Identity",
    "If",
    "IsFinite",
    "LeakyRelu",
    "Less",
    "LessEqual",
    "Log",
    "LogSoftmax",
    "LogicalAnd",
    "LogicalNot",
    "LogicalOr",
    "MatMul",
    "Max",
    "Maximum",
    "Mean",
    "Min",
    "Minimum",
    "Mul",
    "Neg",
    "NoOp",
    "NotEqual",
    "Pack",
    "Pad",
    "Pow",
    "Prod",
    "Range",
    "RealDiv",
    "Reciprocal",
    "Relu",
    "Relu6",
    "ReluGrad",
    "Reshape",
    "Round",
    "Rsqrt",
    "RsqrtGrad",
    "Select",
    "Shape",
    "Sigmoid",
    "SigmoidGrad",
    "Sign",
    "Sin",
    "Size",
    "Slice",
    "Snapshot",
    "Softmax",
    "SoftmaxCrossEntropyWithLogits",
    "Softplus",
    "Split",
    "Sqrt",
    "Square",
    "SquaredDifference",
    "Squeeze",
    "StopGradient",
    "StridedSlice",
    "Sub",
    "Sum",
    "Tanh",
    "TanhGrad",
    "Tile",
    "TopKV2",
    "Transpose",
    "Unpack",
    "While",
    "ZerosLike"
  });

  ops.insert(ops.end(), {
    "AddV2",
    "BatchMatMulV2",
    "GatherV2"
  });

  ops.insert(ops.end(), {
    "SelectV2",
  });

  ops.insert(ops.end(), {
    "Dequantize",
    "QuantizeV2"
  });
  ops.insert(ops.end(), {
    "SparseReshape",
    "SparseFillEmptyRows",
    "SparseSegmentMean",
    "SparseSegmentSum",
    "Where",
  });
  ops.insert(ops.end(), {
    "QuantizedConv2DWithBiasAndRequantize"
  });

  ops.insert(ops.end(), {
    "RandomUniform"
  });

    // clang-format on
    return ops;
}

std::unordered_map<string, std::vector<string>>* GetWhitelistTable()
{
    // Table format: category name: {list of TF operations in that category}
    // clang-format off
    static std::unordered_map<string, std::vector<string>>* result =
        new std::unordered_map<string, std::vector<string>>{
            // Unary
            {"PW",
            {"ComplexAbs", "Angle", "Conj", "Abs", "Acos", "Acosh", "Asin",
              "Atan", "Atanh", "Ceil", "Cos", "Cosh", "Sin", "Exp", "Expm1",
              "Floor", "IsFinite", "IsInf", "IsNan", "Inv", "Reciprocal", "Log",
              "Log1p", "Invert", "LogicalNot", "Ndtri", "Neg", "Rint", "Round",
              "Rsqrt", "Sigmoid", "Sign", "Sinh", "Softplus", "Softsign", "Sqrt",
              "Square", "Tan", "Tanh", "Real", "Imag", "Erf", "Erfc", "Erfinv",
              "Lgamma", "Digamma",
              // Binary
              "Add", "AddV2", "Sub", "Mul", "Div", "Atan2", "Complex", "DivNoNan",
              "MulNoNan", "FloorDiv", "Xlogy", "Xlog1py", "Xdivy", "FloorMod",
              "BitwiseAnd", "BitwiseOr", "BitwiseXor", "LeftShift", "RightShift",
              "LogicalAnd", "LogicalOr", "Mod", "Maximum", "Minimum", "RealDiv",
              "ReciprocalGrad", "RsqrtGrad", "SqrtGrad", "TruncateDiv",
              "TruncateMod", "Equal", "NotEqual", "Greater", "GreaterEqual",
              "Less", "LessEqual", "SigmoidGrad", "SoftplusGrad", "SoftsignGrad",
              "TanhGrad", "Pow", "SquaredDifference", "ApproximateEqual",
              // Others
              "AddN", "Bitcast", "Cast", "ClipByValue", "Const", "Empty",
              "Identity", "IdentityN", "Relu", "Relu6", "ReluGrad", "Relu6Grad",
              "LeakyReluGrad", "Elu", "EluGrad", "Selu", "SeluGrad", "Select",
              "SelectV2", "Transpose", "ConjugateTranspose",
              "_UnaryOpsComposition",
              // The following 4 operations are converted to identity
              "PlaceholderWithDefault", "PreventGradient", "StopGradient",
              "Snapshot"}},
            {"RED",
            {"All", "Any", "Min", "Max", "Mean", "Prod", "Sum"}},
            {"PWRED",
            {"ArgMax", "ArgMin", "DiagPart", "Softmax",
              "SparseSoftmaxCrossEntropyWithLogits", "LogSoftmax"}},
            {"REDUCEWINDOW",
            {"ArgMax", "ArgMin", "DiagPart", "Softmax",
              "SparseSoftmaxCrossEntropyWithLogits", "LogSoftmax"}},
            {"REDUCEWINDOWPW", {"BiasAddGrad", "LRN", "LRNGrad"}},
            {"BN",
            {"FusedBatchNorm", "FusedBatchNormV2", "FusedBatchNormV3",
              "_FusedBatchNormEx", "FusedBatchNormGrad", "FusedBatchNormGradV2",
              "FusedBatchNormGradV3"}},
            {"SORT", {"TopKV2"}},  // XLA version much faster then TF version.
            {"MLIR", GetNpuXlaSupportedOps()},
            {"MISC",
                  {"BroadcastTo", "ExpandDims", "Fill", "NoOp",
        "Range", "Rank", "Reshape", "Shape", "ShapeN", "Size", "Squeeze",
        "Transpose", "ZerosLike", "OnesLike", "BiasAdd" /* PW + Broadcast */,
        "BroadcastArgs", "BroadcastGradientArgs", "OneHot", "Concat", "ConcatV2",
        "ConcatOffset", "Const", "MirrorPad", "Pack", "Pad", "PadV2", "Reverse",
        "ReverseV2", "ReverseSequence", "Slice", "Split", "SplitV",
        "StridedSlice", "StridedSliceGrad", "ResourceStridedSliceAssign",
        "Tile", "Transpose", "InvertPermutation", "Unpack"}}};
    // clang-format on
    return result;
}

namespace testing {
void ResetClusterSequenceNumber()
{
    g_clusterSequenceNum = 0;
}

absl::flat_hash_set<string> GetKnownXLAAllowlistOp()
{
    absl::flat_hash_set<string> result{"AdjustContrastv2",
                                       "AdjustHue",
                                       "AdjustSaturation",
                                       "Asinh",
                                       "Assert",
                                       "AssignAddVariableOp",
                                       "AssignSubVariableOp",
                                       "AssignVariableOp",
                                       "AssignVariableXlaConcatND",
                                       "AvgPool",
                                       "AvgPool3D",
                                       "AvgPool3DGrad",
                                       "AvgPoolGrad",
                                       "BatchMatMul",
                                       "BatchMatMulV2",
                                       "BatchMatMulV3",
                                       "BatchToSpace",
                                       "BatchToSpaceND",
                                       "BesselI0e",
                                       "BesselI1e",
                                       "Betainc",
                                       "BiasAddV1",
                                       "Bincount",
                                       "Bucketize",
                                       "Case",
                                       "CheckNumerics",
                                       "Cholesky",
                                       "ControlTrigger",
                                       "Conv",
                                       "Conv2D",
                                       "Conv2DBackpropFilter",
                                       "Conv2DBackpropInput",
                                       "Conv3D",
                                       "Conv3DBackpropFilterV2",
                                       "Conv3DBackpropInputV2",
                                       "Cross",
                                       "Cumprod",
                                       "Cumsum",
                                       "CumulativeLogsumexp",
                                       "DenseBincount",
                                       "DataFormatDimMap",
                                       "DataFormatVecPermute",
                                       "DepthToSpace",
                                       "DepthwiseConv2dNative",
                                       "DepthwiseConv2dNativeBackpropFilter",
                                       "DepthwiseConv2dNativeBackpropInput",
                                       "Dequantize",
                                       "Diag",
                                       "DynamicInfeedEnqueueTupleOp",
                                       "DynamicInfeedDequeueTupleOp",
                                       "DynamicStitch",
                                       "DynamicPartition",
                                       "Einsum",
                                       "EmptyTensorList",
                                       "EnsureShape",
                                       "ExtractImagePatches",
                                       "Igamma",
                                       "IgammaGradA",
                                       "RandomGammaGrad",
                                       "Igammac",
                                       "FFT",
                                       "FFT2D",
                                       "FFT3D",
                                       "FakeParam",
                                       "FakeQuantWithMinMaxArgs",
                                       "FakeQuantWithMinMaxArgsGradient",
                                       "FakeQuantWithMinMaxVars",
                                       "FakeQuantWithMinMaxVarsGradient",
                                       "FakeQuantWithMinMaxVarsPerChannel",
                                       "FakeQuantWithMinMaxVarsPerChannelGradient",
                                       "Gather",
                                       "GatherNd",
                                       "GatherV2",
                                       "HSVToRGB",
                                       "IFFT",
                                       "IFFT2D",
                                       "IFFT3D",
                                       "IRFFT",
                                       "IRFFT2D",
                                       "IRFFT3D",
                                       "If",
                                       "InTopKV2",
                                       "L2Loss",
                                       "LeakyRelu",
                                       "LinSpace",
                                       "ListDiff",
                                       "LogMatrixDeterminant",
                                       "LowerBound",
                                       "MatMul",
                                       "MatrixBandPart",
                                       "MatrixDiag",
                                       "MatrixDiagPart",
                                       "MatrixDiagPartV2",
                                       "MatrixDiagPartV3",
                                       "MatrixDiagV2",
                                       "MatrixDiagV3",
                                       "MatrixInverse",
                                       "MatrixSetDiag",
                                       "MatrixSetDiagV2",
                                       "MatrixSetDiagV3",
                                       "MatrixSolve",
                                       "MatrixTriangularSolve",
                                       "MaxPool",
                                       "MaxPool3D",
                                       "MaxPool3DGrad",
                                       "MaxPool3DGradGrad",
                                       "MaxPoolGrad",
                                       "MaxPoolGradGrad",
                                       "MaxPoolGradGradV2",
                                       "MaxPoolGradV2",
                                       "MaxPoolV2",
                                       "Multinomial",
                                       "NextAfter",
                                       "NonMaxSuppressionV3",
                                       "NonMaxSuppressionV4",
                                       "ParallelDynamicStitch",
                                       "ParameterizedTruncatedNormal",
                                       "PartitionedCall",
                                       "Polygamma",
                                       "PopulationCount",
                                       "Qr",
                                       "QuantizeAndDequantizeV2",
                                       "QuantizeAndDequantizeV3",
                                       "QuantizeAndDequantizeV4",
                                       "RFFT",
                                       "RFFT2D",
                                       "RFFT3D",
                                       "RGBToHSV",
                                       "RandomShuffle",
                                       "RandomStandardNormal",
                                       "RandomUniform",
                                       "RandomUniformInt",
                                       "ReadVariableOp",
                                       "ReadVariableXlaSplitND",
                                       "ResizeBilinear",
                                       "ResizeBilinearGrad",
                                       "ResizeNearestNeighbor",
                                       "ResourceApplyAdaMax",
                                       "ResourceApplyAdadelta",
                                       "ResourceApplyAdagrad",
                                       "ResourceApplyAdagradDA",
                                       "ResourceApplyAdagradV2",
                                       "ResourceApplyAdam",
                                       "ResourceApplyAddSign",
                                       "ResourceApplyCenteredRMSProp",
                                       "ResourceApplyFtrl",
                                       "ResourceApplyFtrlV2",
                                       "ResourceApplyGradientDescent",
                                       "ResourceApplyKerasMomentum",
                                       "ResourceApplyMomentum",
                                       "ResourceApplyPowerSign",
                                       "ResourceApplyProximalAdagrad",
                                       "ResourceApplyProximalGradientDescent",
                                       "ResourceApplyRMSProp",
                                       "ResourceGather",
                                       "ResourceScatterAdd",
                                       "ResourceScatterDiv",
                                       "ResourceScatterMax",
                                       "ResourceScatterMin",
                                       "ResourceScatterMul",
                                       "ResourceScatterNdAdd",
                                       "ResourceScatterNdSub",
                                       "ResourceScatterNdUpdate",
                                       "ResourceScatterSub",
                                       "ResourceScatterUpdate",
                                       "RngReadAndSkip",
                                       "RngSkip",
                                       "Roll",
                                       "ScatterNd",
                                       "SegmentSumV2",
                                       "SegmentProdV2",
                                       "SegmentMinV2",
                                       "SegmentMaxV2",
                                       "SelfAdjointEigV2",
                                       "SoftmaxCrossEntropyWithLogits",
                                       "SpaceToBatch",
                                       "SpaceToBatchND",
                                       "SpaceToDepth",
                                       "SparseMatMul",
                                       "SparseToDense",
                                       "StackCloseV2",
                                       "StackPopV2",
                                       "StackPushV2",
                                       "StackV2",
                                       "StatefulPartitionedCall",
                                       "StatefulStandardNormalV2",
                                       "StatefulTruncatedNormal",
                                       "StatefulUniform",
                                       "StatefulUniformFullInt",
                                       "StatefulUniformInt",
                                       "StatelessCase",
                                       "StatelessIf",
                                       "StatelessMultinomial",
                                       "StatelessParameterizedTruncatedNormal",
                                       "StatelessRandomGetAlg",
                                       "StatelessRandomGetKeyCounter",
                                       "StatelessRandomGetKeyCounterAlg",
                                       "StatelessRandomNormal",
                                       "StatelessRandomNormalV2",
                                       "StatelessRandomUniform",
                                       "StatelessRandomUniformV2",
                                       "StatelessRandomUniformInt",
                                       "StatelessRandomUniformIntV2",
                                       "StatelessRandomUniformFullInt",
                                       "StatelessRandomUniformFullIntV2",
                                       "StatelessTruncatedNormal",
                                       "StatelessTruncatedNormalV2",
                                       "StatelessWhile",
                                       "StochasticCastToInt",
                                       "Svd",
                                       "SymbolicGradient",
                                       "TensorArrayCloseV3",
                                       "TensorArrayConcatV3",
                                       "TensorArrayGatherV3",
                                       "TensorArrayGradV3",
                                       "TensorArrayReadV3",
                                       "TensorArrayScatterV3",
                                       "TensorArraySizeV3",
                                       "TensorArraySplitV3",
                                       "TensorArrayV3",
                                       "TensorArrayWriteV3",
                                       "TensorListConcatV2",
                                       "TensorListElementShape",
                                       "TensorListFromTensor",
                                       "TensorListGather",
                                       "TensorListGetItem",
                                       "TensorListLength",
                                       "TensorListPopBack",
                                       "TensorListPushBack",
                                       "TensorListReserve",
                                       "TensorListSetItem",
                                       "TensorListSplit",
                                       "TensorListStack",
                                       "TensorScatterAdd",
                                       "TensorScatterMax",
                                       "TensorScatterMin",
                                       "TensorScatterSub",
                                       "TensorScatterUpdate",
                                       "ToBool",
                                       "TridiagonalSolve",
                                       "TridiagonalMatMul",
                                       "TruncatedNormal",
                                       "UniformDequantize",
                                       "UniformQuantize",
                                       "UniformQuantizedAdd",
                                       "UniformQuantizedClipByValue",
                                       "UniformQuantizedConvolution",
                                       "UniformQuantizedDot",
                                       "UniformRequantize",
                                       "Unique",
                                       "UniqueV2",
                                       "UpperBound",
                                       "UnsortedSegmentMax",
                                       "UnsortedSegmentMin",
                                       "UnsortedSegmentProd",
                                       "UnsortedSegmentSum",
                                       "VarIsInitializedOp",
                                       "VariableShape",
                                       "Where",
                                       "While",
                                       "XlaAllReduce",
                                       "XlaBroadcastHelper",
                                       "XlaCallModule",
                                       "XlaConcatND",
                                       "XlaConv",
                                       "XlaConvV2",
                                       "XlaCustomCall",
                                       "XlaCustomCallV2",
                                       "XlaDequantize",
                                       "XlaDot",
                                       "XlaDotV2",
                                       "XlaDynamicSlice",
                                       "XlaDynamicUpdateSlice",
                                       "XlaEinsum",
                                       "XlaGather",
                                       "XlaIf",
                                       "XlaKeyValueSort",
                                       "XlaOptimizationBarrier",
                                       "XlaPad",
                                       "XlaRecv",
                                       "XlaReduce",
                                       "XlaReducePrecision",
                                       "XlaReduceScatter",
                                       "XlaReduceWindow",
                                       "XlaRemoveDynamicDimensionSize",
                                       "XlaReplicaId",
                                       "XlaRngBitGenerator",
                                       "XlaScatter",
                                       "XlaSelectAndScatter",
                                       "XlaSelfAdjointEig",
                                       "XlaSend",
                                       "XlaSetBound",
                                       "XlaSetDynamicDimensionSize",
                                       "XlaSharding",
                                       "XlaSort",
                                       "XlaSplitND",
                                       "XlaSpmdFullToShardShape",
                                       "XlaSpmdShardToFullShape",
                                       "XlaSvd",
                                       "XlaVariadicReduce",
                                       "XlaVariadicReduceV2",
                                       "XlaVariadicSort",
                                       "XlaWhile",
                                       "Zeta",
                                       "_Arg",
                                       "_ArrayToList",
                                       "_ListToArray",
                                       "_Retval"};
    return result;
}

}  // namespace testing
}  // namespace npu_xla
}  // namespace tensorflow
