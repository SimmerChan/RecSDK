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

#include "tf_bridge/passes/encapsulate_subgraphs_pass.h"

#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tf_bridge/common.h"
#include "tf_bridge/passes/defunctionalize_control_flow.h"
#include "tf_bridge/passes/mark_for_npu_compilation_pass.h"
#include "tf_bridge/tf/compilability_check_util.h"
#include "tf_bridge/tf/const_analysis.h"
#include "tf_bridge/tf/errors.h"
#include "tf_bridge/tf/graphcycles.h"
#include "tf_bridge/tf/shape_inference_helpers.h"
#include "tf_bridge/utils/dump_graph.h"

namespace tensorflow {
namespace npu_xla {

const char* const kXlaCompiledKernelAttr = "_NpuXlaCompiledKernel";
const char* const kXlaNumConstantArgsAttr = "_NpuXlaNumConstantArgs";
const char* const kXlaNumResourceArgsAttr = "_NpuXlaNumResourceArgs";
const char* const kMlirNumFixedShapeArgsAttr = "_NpuMlirNumFixedShapeArgs";
const char* const kMlirNumHostArgsAttr = "_NpuMlirNumHostArgs";
const char* const kMlirNumHostRetsAttr = "_NpuMlirNumHostRets";
const char* const kXlaHostTransferSequencerAttr = "_npu_xla_host_transfer_sequencer";

void SortControlInputs(GraphDef* gdef)
{
    int64 num_nodes = gdef->node_size();
    for (int64 i = 0; i < num_nodes; ++i) {
        NodeDef* node = gdef->mutable_node(i);
        // Stable sort control inputs and leave the order of data inputs unchanged.
        std::stable_sort(node->mutable_input()->begin(), node->mutable_input()->end(),
                         [](const string& a, const string& b) {
                             bool a_is_control = absl::StartsWith(a, "^");
                             bool b_is_control = absl::StartsWith(b, "^");
                             return (!a_is_control && b_is_control) || (a_is_control && b_is_control && a < b);
                         });
    }
}

namespace {

bool AreAllParentsGuaranteedConst(const Node& n, const std::unordered_set<const Node*>& runtime_const_nodes)
{
    if (n.type_string() == "GuaranteeConst") {
        // If the current node is itself a cast-to-const, no need
        // to look at the incoming edges.
        return true;
    }

    bool allParentsConst = true;
    bool atleastOneNonControlEdge = false;
    for (const Edge* in : n.in_edges()) {
        atleastOneNonControlEdge = atleastOneNonControlEdge || !in->IsControlEdge();
        if (!in->IsControlEdge() && runtime_const_nodes.count(in->src()) == 0) {
            allParentsConst = false;
            break;
        }
    }
    return allParentsConst && atleastOneNonControlEdge;
}

void MarkGuaranteedConstants(const Graph& graph, const std::vector<std::pair<const Node*, Node*>>& src_arg_pairs)
{
    std::unordered_set<const Node*> guaranteed_const_nodes;
    std::vector<const Node*> srcs;
    srcs.reserve(src_arg_pairs.size());
    for (const auto& src_arg : src_arg_pairs) {
        srcs.push_back(src_arg.first);
    }
    ReverseDFSFrom(graph, srcs, nullptr, [&guaranteed_const_nodes](const Node* n) {
        // Doesn't work in the presence of loops.
        if (AreAllParentsGuaranteedConst(*n, guaranteed_const_nodes)) {
            guaranteed_const_nodes.insert(n);
        }
    });

    for (auto& src_arg : src_arg_pairs) {
        if (guaranteed_const_nodes.count(src_arg.first) != 0) {
            VLOG(1) << "Guaranteed const found: " << src_arg.first->DebugString();
            src_arg.second->AddAttr("_is_guaranteed_constant", true);
        }
    }
}

struct OutputInputTensorPairHasher {
    uint64 operator()(std::pair<OutputTensor, InputTensor> const& s) const
    {
        return Hash64Combine(OutputTensor::Hash()(s.first), InputTensor::Hash()(s.second));
    }
};

// add a canonical copy of these operator names and refactor
// everything to use it.
static const char* const kArgOp = "_Arg";
static const char* const kRetValOp = "_Retval";
static const char* const kHostComputeOp = "XlaHostCompute";
static const char* const kSendFromHostOp = "_XlaSendFromHost";
static const char* const kRecvAtHostOp = "_XlaRecvAtHost";

class Encapsulator {
public:
    Encapsulator(string group_attribute, string outside_compilation_attribute, Graph const* graph_in)
        : group_attribute_(std::move(group_attribute)),
          outside_compilation_attribute_(std::move(outside_compilation_attribute)),
          graph_in_(graph_in)
    {
    }

    // Find dependencies between subgraphs and outside_compilation clusters that
    // only manifest via edges between outside_compilation clusters in the outer
    // (non-compiled) graph.
    Status FindClusterDependencies();

    // Find subgraphs marked with 'group_attribute', and build a new
    // subgraph, one for each value of 'group_attribute'.
    Status SplitIntoSubgraphs(FunctionLibraryDefinition* library);

    // Build a FunctionDef for each subgraph, and add it 'library'. The values of
    // the 'group_attribute' annotations become the function names.
    // If 'reuse_existing_functions' is set, use an existing function with the
    // same name, if any.
    // If 'rewrite_subgraph_fn' is set, it is applied to each subgraph before
    // function conversion.
    Status BuildFunctionDefs(const RewriteSubgraphFn& rewrite_subgraph_fn, bool reuse_existing_functions,
                             FunctionLibraryDefinition* library);

    // Write a copy of the input graph to 'graph_out', where the subgraphs are
    // replaced with calls to the new functions.
    Status BuildOutputGraph(Graph* graph_out, FunctionLibraryDefinition* library);

private:
    // A subgraph of the input, all marked with a common 'group_attribute'
    // value. A subgraph may contain multiple `outside_compilation' clusters.
    //
    // In the following simple example, A, B, ..., E are nodes in the original
    // graph. The group attributes and outside_compilation attributes g and oc are
    // each shown as either 0 or empty.
    //
    //  A  -->  B  -->  C  -->  D  -->  E
    //  g:      g:0     g:0     g:0     g:
    //  oc:     oc:     oc:0    oc:     oc:
    //
    // The example is rewritten to two graphs; one on the host and one to be
    // compiled. The host graph is as follows. RAH is a RecvAtHost node receiving
    // input from the compiled cluster, and SFH is a SendFromHost node sending
    // input back to the compiled cluster. Dotted edges are control edges. A
    // 'sequencing' node S is inserted, and both RAH and SFH are connected via S
    // to E (and in general all nodes that depend on nodes in the compiled
    // cluster) to ensure that they are not pruned.
    //
    //  A  -->  Call  -->  E
    //                     ^
    //                     .
    //           ........> S
    //       ....          ^
    //     ..             .
    //  RAH -->  C  --> SFH
    //
    // The compiled cluster is as follows. HC is a HostCompute node which is the
    // source of a channel to the RAH node above and the destination of a channel
    // from the SFH node above.
    //
    //  Arg  --> B  --> HC  --> D --> Retval
    //
    // The channels HC/RAH and SFH/HC each transmit multiple tensors, so there is
    // at most one RAH and SFH in each outside_compilation cluster. This design is
    // preferred over adding separate Arg/Retval nodes for each transmitted value
    // because it allows optimizations to the host code that would like to limit
    // communication between host and device and, e.g., raise only one interrupt
    // per channel rather than one per transmitted value.
    //
    // The shapes of the outputs from the HC node in general cannot be determined
    // until the shapes of its inputs are known at compile time, since e.g.,
    // above, the shape of C's outputs aren't known until the shape of its inputs
    // are known. If the shapes of the HC's outputs can be determined during the
    // rewrite, they are stored in the node's 'shapes' attr. Otherwise a minimal
    // graph is stored in the shape_inference_graph attr. This graph can be used
    // when compiling the HC Op to determined the shape of the SFH inputs given
    // the shapes of any ancestor RAH outputs. If it can be determined that the
    // shape of the SFH inputs will not be inferrable even once the shapes of the
    // RAH outputs are known, an error is returned by the rewriter.
    //
    // Once edges between compiled and outside_compilation clusters have been
    // replaced by send/recv ops, some dependencies may no longer be apparent.
    // A clustering pass finds all the dependencies between HC nodes that are only
    // present as a result of edges between nodes in outside_compilation clusters.
    // Suppose there is a path from outside_compilation cluster C in subgraph S
    // to outside_compilation cluster D in subgraph T. If S != T then a control
    // edge is added from the call node for S to the call node for T, which
    // ensures that C will execute before D because S executes before T. If S==T
    // then a control dependency is added between the HC nodes for C and D in S,
    // and the HC node for C is added to an 'ancestors' attr in the HC node for D
    // so that during compilation of the HC node for D, an XLA control dependency
    // can be added to ensure C's SendToHost executes before D's RecvFromHost.
    class Subgraph {
    public:
        // Creates a graph to build the subgraph in, if it doesn't already exist,
        // using the same op registry and versions as graph_in.
        Node* MakeNodeImage(const Graph* graph_in, Node* node);

        // Returns the graph the subgraph is being built in.
        Graph* GetGraph() const;

        // Builds a FunctionDef, and adds it to 'library'. The value of the
        // 'group_attribute' annotations becomes the function name.  If
        // 'reuse_existing_functions' is set, use an existing function with the same
        // name, if any.  If 'rewrite_subgraph_fn' is set, it is applied to the
        // subgraph before function conversion.
        Status BuildFunctionDef(const string& name_in, const RewriteSubgraphFn& rewrite_subgraph_fn,
                                bool reuse_existing_functions, FunctionLibraryDefinition* library);
        Status AddShapeToFunctionDef(Graph& graph, FunctionDef* fdef);

        // Adds the function call node to graph_out.
        Status AddFunctionCallNode(const std::unordered_map<const Node*, Node*>& node_images, Graph* graph_out);

        // Adds _RecvAtHost and _SendFromHost nodes, where needed, to graph_out.
        Status AddOutsideCompilationHostIONodes(const string& group_attribute, const string& subgraph_name,
                                                const string& outside_compilation_attribute,
                                                const std::unordered_map<const Node*, Node*>& node_images,
                                                Graph* graph_out);

        // Returns the names of all the outside_compilation subgraphs in this
        // Subgraph.
        void GetOutsideCompilationSubgraphNames(std::vector<string>* names) const;

        // Returns the Node that the inputs and outputs of the function should be
        // wired up to.
        Node* GetCallNode() const;

        // Returns the index of the arg that the dst of edge should connect to.
        int GetArgIndexForEdge(const Edge* edge) const;

        // Returns the index of the result that the src of edge should connect to.
        int GetResultIndexForEdge(const Edge* edge) const;

        // Returns the RecvAtHost node for an outside_compilation subgraph.
        Node* GetRecvAtHostNode(const string& outside_compilation_subgraph_name) const;

        // Returns the output slot for the RecvAtHost node that corresponds to the
        // source of edge in an outside_compilation subgraph.
        int GetRecvAtHostSlot(const string& outside_compilation_subgraph_name, const Edge* edge) const;

        // Returns the SendFromHost node for an outside_compilation subgraph.
        Node* GetSendFromHostNode(const string& outside_compilation_subgraph_name) const;

        // Returns the input slot for the SendFromHost node that corresponds to the
        // destination of edge in an outside_compilation subgraph.
        int GetSendFromHostSlot(const string& outside_compilation_subgraph_name, const Edge* edge) const;

        // Creates an _Arg node for the src node of edge, and add its index to
        // args_by_src_, if none exists yet. Also adds its index to args_by_dst_,
        // and adds the edge within the subgraph from the _Arg node to the image of
        // the dst node.
        Status RecordArg(const Edge* edge, const std::unordered_map<const Node*, Node*>& node_images,
                         std::vector<std::pair<const Node*, Node*>>* src_arg_pairs);

        // Creates a _Retval node for the src node of edge, and add it to results_,
        // if none exists yet. If a new _Retval node is created, also adds the edge
        // within the subgraph from the src to the _Retval node.
        Status RecordResult(const Edge* edge, const std::unordered_map<const Node*, Node*>& node_images);

        // Creates an outside_compilation subgraph for outside_compilation_id if
        // none exists yet. Creates an entry for the src node of edge in the list of
        // inputs for the outside_compilation subgraph, if none exists yet.
        void RecordOutsideCompilationInputOrControl(const string& outside_compilation_id, const Edge* edge);

        // Creates an outside_compilation subgraph for outside_compilation_id if
        // none exists yet. Creates an entry for the src node of edge in the list of
        // outputs by src for the outside_compilation subgraph, if none exists
        // yet. Creates an entry for the dst node of edge in the list of outputs by
        // dst for the outside_compilation subgraph.
        void RecordOutsideCompilationOutputOrControl(const string& outside_compilation_id, const Edge* edge);

        // Records the fact that there is a path from a node in outside_compilation
        // cluster ancestor to node in cluster successor that does not go through
        // the subgraph.
        void RecordOutsideCompilationDependency(const string& successor, const string& ancestor);

        // Returns the mapping from outside_compilation cluster C to the set of
        // outside_compilation clusters that have a path to C entirely outside
        // compiled subgraphs.
        const std::unordered_map<string, std::unordered_set<string>> OutsideCompilationAncestorMap() const;

        // Adds the HostCompute nodes for each outside_compilation subgraph.
        Status AddHostComputes(const string& subgraph_name, const std::unordered_map<const Node*, Node*>& node_images);

        // Creates the sequencer node if it doesn't exist, adding it to graph_out.
        Status MakeSequencingNode(const string& subgraph_name, Graph* graph_out);

        // If there is a sequencer node, adds a control edge from the sequencer to
        // the call node.
        void ConnectSequencerToCallNode(Graph* graph_out);

        Status AddShapeInferenceInfo(const string& subgraph_name, const string& outside_compilation_subgraph_name,
                                     const std::vector<TensorShapeProto>& shapes, Graph* inference_graph,
                                     FunctionLibraryDefinition* library);

        Status ReplaceFunctionDef(FunctionLibraryDefinition* library);

        // Detect and lower while and if op to control flow primitives
        StatusOr<bool> MaybeDefunctionalize(FunctionLibraryDefinition* library);

    private:
        struct OutsideCompilationSubgraph {
            // Map from source (producer node/slot) tensors in the original graph to
            // input index (slot number in the HostCompute/RecvAtHost nodes that will
            // be created) for the outside_compilation subgraph.
            std::unordered_map<OutputTensor, int, OutputTensor::Hash> inputs;

            // Set of nodes in the original graph that are the source of control edges
            // that cross from the containing compiled subgraph into the
            // outside_compilation subgraph. These are recorded by
            // RecordOutsideCompilationInputOrControl while walking all the subgraph
            // edges, and lifted control edges within the subgraph are added by
            // AddSendsToOutsideCompilation once the _HostCompute node has been
            // created. The matching control edge from _RecvAtHost to the
            // destination is added by CopyEdgeToOutputGraph.
            std::unordered_set<const Node*> control_inputs;

            // Maps from source (producer node/slot) and destination (consumer
            // node/slot) tensors in the original graph to output index (slot number
            // in the SendFromHost/HostCompute nodes that will be created) for the
            // outside_compilation subgraph.
            struct ArgNumAndType {
                int index;
                DataType dtype;

                ArgNumAndType(int i, DataType t) : index(i), dtype(t) {}
            };
            std::unordered_map<OutputTensor, ArgNumAndType, OutputTensor::Hash> outputs_by_src;
            std::unordered_map<InputTensor, int, InputTensor::Hash> outputs_by_dst;

            // Set of nodes in the original graph that are the destination of control
            // edges that cross from the outside_compilation subgraph into the
            // containing compiled subgraph. These are recorded by
            // RecordOutsideCompilationOutputOrControl while walking all the subgraph
            // edges, and lifted control edges within the subgraph are added by
            // AddRecvsFromToOutsideCompilation once the _HostCompute node has been
            // created. The matching control edge from the source to _SendFromHost to
            // the destination is added by CopyEdgeToOutputGraph.
            std::unordered_set<const Node*> control_outputs;

            // Name of the _HostCompute node in the subgraph.
            string host_compute_name;

            // _RecvAtHost node in the output graph. Not owned.
            Node* recv_at_host = nullptr;

            // _SendFromHost node in the output graph. Not owned.
            Node* send_from_host = nullptr;
        };

        // Creates an outside_compilation subgraph for outside_compilation_id if
        // none exists yet. Returns the (possible newly created) subgraph for
        // outside_compilation_id.
        OutsideCompilationSubgraph* LookupOrCreateOutsideCompilationSubgraph(const string& outside_compilation_id);

        // Builds a placeholder node used to provide the key input to a RecvAtHost
        // or SendFromHost node. This placeholder node will be removed by a later
        // pass.
        Status AddHostComputeKeyPlaceholder(OutsideCompilationSubgraph* oc_subgraph, Graph* graph_out);

        // Get the set of outside_compilation clusters and the dependency edges
        // between them.
        void GetActiveClusterDependencyGraph(std::unordered_set<string>* clusters,
                                             std::unordered_set<string>* has_successor,
                                             std::unordered_map<string, std::unordered_set<string>>* ancestors_map);

        // Builds a _RecvAtHost node producing all the inputs of an
        // outside_compilation subgraph and stores it in oc_subgraph.recv_at_host.
        Status AddRecvAtHostNode(const string& group_attribute, const string& subgraph_name,
                                 const string& outside_compilation_attribute, const string& oc_subgraph_name,
                                 OutsideCompilationSubgraph* oc_subgraph, Graph* graph_out);

        // Builds a _SendFromHost node consuming all the outputs of an
        // outside_compilation subgraph and stores it in oc_subgraph.send_from_host.
        Status AddSendFromHostNode(const std::unordered_map<const Node*, Node*>& node_images,
                                   const string& group_attribute, const string& subgraph_name,
                                   const string& outside_compilation_attribute, const string& oc_subgraph_name,
                                   OutsideCompilationSubgraph* oc_subgraph, Graph* graph_out);

        // The subgraph extracted from the input graph, suitable for being turned
        // into a FunctionDef. Inputs are fed by _Arg nodes, and outputs are
        // returned by _Retval nodes.
        std::unique_ptr<Graph> graph_;

        // Which device are these nodes on? Used to assign a device to the call
        // node.
        string device_;

        // NodeDef for the function call node.
        NodeDef call_node_def_;

        // Name that is used for the call node. This may not be
        // call_node_def_.name() if the client supplies a rewrite lambda.
        string function_def_name_;

        // Placeholder node simulating the host compute key in the output graph.
        // Not owned.
        Node* host_compute_key_placeholder_ = nullptr;

        // Function call node in the output graph. Not owned.
        Node* call_node_;

        // Maps from source (producer node/slot) and destination
        // (consumer node/slot) tensors in the input graph to _Arg numbers in
        // the subgraph. The source map is one-to-one, whereas the dest map may be
        // many-to-one.
        std::unordered_map<OutputTensor, int, OutputTensor::Hash> args_by_src_;
        std::unordered_map<InputTensor, int, InputTensor::Hash> args_by_dst_;

        // The arguments to the subgraph, in order.
        std::vector<Node*> args_;

        // Map from source tensor in the input graph to result #.
        std::unordered_map<OutputTensor, int, OutputTensor::Hash> results_;

        // The outside_compilation clusters in this subgraph.
        std::unordered_map<string, OutsideCompilationSubgraph> outside_compilation_subgraphs_;
        // For each outside_compilation cluster C, the outside_compilation clusters
        // that have a path to C outside the compiled graph.
        std::unordered_map<string, std::unordered_set<string>> outside_compilation_ancestors_;
        // For each outside_compilation cluster C, the outside_compilation clusters
        // that have a path from C outside the compiled graph.
        std::unordered_map<string, std::unordered_set<string>> outside_compilation_successors_;

        // NoOp node in the output graph that is sequenced after the call node and
        // used to prevent host-side outside_compilation sends and recvs from being
        // pruned.
        Node* sequencer_ = nullptr;
    };

    // Returns the key attribute and outside_compilation attribute associated
    // with a node in attr, and outside_compilation_attr, respectively. Sets
    // either result to the empty string if the respective attribute is not
    // found. Returns error status if there is an outside_compilation attribute
    // and no key attribute,
    Status GetFunctionNameAttr(Node const* node, string* attr, string* outside_compilation_attr) const;

    // Copies edges local to a subgraph. Adds _Arg and _Retval nodes to
    // subgraphs for data edges that cross subgraph boundaries.
    Status CopySubgraphEdges(const std::unordered_map<const Node*, Node*>& node_images,
                             std::vector<std::pair<const Node*, Node*>>* src_arg_pairs);

    // Copies all marked nodes to a subgraph. Does nothing for unmarked nodes,
    // or nodes marked outside_compilation.
    Status CopySubgraphNodes(std::unordered_map<const Node*, Node*>* node_images);

    // Copies all nodes that aren't in a compiled subgraph to the output graph.
    Status CopyNodesToOutputGraph(Graph* graph_out, std::unordered_map<const Node*, Node*>* node_images);

    // Adds function call nodes for each compiled subgraph.
    Status AddFunctionCallNodes(const std::unordered_map<const Node*, Node*>& node_images, Graph* graph_out);

    // Adds _RecvAtHost and _SendFromHost nodes, where needed, for all
    // outside_compilation subgraphs.
    Status AddOutsideCompilationHostIONodes(const std::unordered_map<const Node*, Node*>& node_images,
                                            Graph* graph_out);

    // Finds the image of an edge source in the output graph. If the edge crosses
    // a subgraph boundary it is the output of a call node, otherwise it is a node
    // in the output graph.
    Status FindOutputImageOfEdgeSrc(const string& src_func_id, const string& src_outside_compilation_id,
                                    const string& dst_func_id, const string& dst_outside_compilation_id,
                                    const std::unordered_map<const Node*, Node*>& node_images,
                                    const Node* original_src_node, Node** src_image);

    // Finds an edge source slot in the output graph. If the edge crosses a
    // subgraph boundary it is a slot on the output of a call node or a
    // _RecvAtHost node, otherwise it is a slot on a node in the output graph.
    int FindOutputSlotOfEdgeSrc(const string& src_func_id, const string& src_outside_compilation_id,
                                const string& dst_func_id, const string& dst_outside_compilation_id, const Edge* edge);

    // Finds the image of an edge destination in the output graph. If the edge
    // crosses a subgraph boundary it is the input of a call node or a
    // _SendFromHost node, otherwise it is a node in the output graph.
    Status FindOutputImageOfEdgeDst(const string& src_func_id, const string& src_outside_compilation_id,
                                    const string& dst_func_id, const string& dst_outside_compilation_id,
                                    const std::unordered_map<const Node*, Node*>& node_images,
                                    const Node* original_dst_node, Node** dst_image);

    // Finds an edge destination slot in the output graph. If the edge crosses a
    // subgraph boundary it is a slot on the input of a call node or a
    // _SendFromHost node, otherwise it is a slot on a node in the output graph.
    int FindOutputSlotOfEdgeDst(const string& src_func_id, const string& src_outside_compilation_id,
                                const string& dst_func_id, const string& dst_outside_compilation_id, const Edge* edge);

    // Copies a single edge to the output graph. The edge is either entirely
    // within the output graph, or crosses into or out of a compiled subgraph.
    Status CopyEdgeToOutputGraph(
        const Edge* edge, const string& src_func_id, const string& src_outside_compilation_id,
        const string& dst_func_id, const string& dst_outside_compilation_id,
        const std::unordered_map<const Node*, Node*>& node_images, Graph* graph_out,
        std::unordered_set<std::pair<OutputTensor, InputTensor>, OutputInputTensorPairHasher>* edges_added);

    // Adds control dependencies between subgraph call nodes that have
    // dependencies via outside_compilation edges.
    Status AddCallNodeDependencies(Graph* graph_out);

    // Adds all edges to the output graph.
    Status AddEdgesToOutputGraph(const std::unordered_map<const Node*, Node*>& node_images, Graph* graph_out);

    // Constructs a minimal shape inference graph that can be used to determine
    // the shape of send_node at the time that the subgraph is compiled.
    // recv_at_host_nodes contains the names of all the recv_at_host nodes that
    // send_node might depend on. These recv_at_host nodes have shapes that are
    // not known during the rewrite pass, but will be known at compile time.
    // If the shapes of all the inputs to send_node can be determined during the
    // rewrite pass, on exit graphdef_out is empty and the shapes are returned in
    // static_shape_out. Otherwise graphdef_out contains a graph that can be used
    // for shape inference at compile time, where all the source nodes of the
    // graph are either constants with known shapes, or nodes named in
    // recv_at_host_nodes.
    // A non-OK status is returned if neither of the above conditions can be
    // satisfied, e.g., because send_node depends on a node that doesn't have a
    // registered shape inference function.
    Status DoStaticShapeInferenceForOutsideCompilationSend(
        const Graph& graph_in, const BackEdgeHelper& back_edge_helper, const ShapeRefiner& shape_refiner,
        const std::unordered_set<string>& recv_at_host_nodes, Node* send_node, FunctionLibraryDefinition* library,
        std::vector<TensorShapeProto>* static_shape_out, std::unique_ptr<Graph>* graph_out);

    // Makes a copy of graph containing only nodes that are ancestors of at least
    // one node in send_from_host_nodes and store it in pruned_graph. On exit
    // nodes_images contains a mapping from nodes in graph to nodes in
    // pruned_graph. All functions in the copied graph are inlined.
    Status MakePrunedGraphCopyAndInline(const Graph& graph, const std::vector<Node*>& sink_nodes,
                                        std::unique_ptr<Graph>* pruned_graph,
                                        std::unordered_map<const Node*, Node*>* node_images,
                                        FunctionLibraryDefinition* library);

    // Makes a copy of graph containing only nodes that are ancestors of a
    // send_from_host node in an outside_compilation subgraph, and store it in
    // pruned_graph. Also perform shape inference on the pruned graph, using
    // shape_refiner. On exit node_images contains a mapping from nodes in graph
    // to nodes in pruned_graph.
    Status MakeGraphForOutsideCompilationSends(const Graph& graph, std::unique_ptr<Graph>* pruned_graph,
                                               BackEdgeHelper* back_edge_helper, ShapeRefiner* shape_refiner,
                                               std::unordered_map<const Node*, Node*>* node_images,
                                               FunctionLibraryDefinition* library);

    // Performs static shape inference, as far as possible, for the send_from_host
    // nodes in each outside_compilation subgraph. Where it is not possible to
    // determine the shape statically, stores a serialized GraphDef in the
    // HostCompute 'shape_inference_graph' attr, to be used at compile time for
    // final inference. If the shapes are known statically they are stored in the
    // HostCompute 'shapes' attr.
    Status GetShapeInfoForOutsideCompilationSends(Graph* graph_out, FunctionLibraryDefinition* library);

    const string group_attribute_;
    const string outside_compilation_attribute_;
    const Graph* graph_in_;

    std::unordered_map<string, Subgraph> subgraphs_;
    // For each subgraph S the subgraphs S' such that there is a path in some
    // outside_compilation cluster C in S to some outside_compilation cluster C'
    // in S', that goes only through the uncompiled graph.
    std::unordered_map<string, std::unordered_set<string>> subgraph_ancestors_;

    TF_DISALLOW_COPY_AND_ASSIGN(Encapsulator);
};

namespace {

// Return in 'sorted' a topological sort of clusters according to the
// dependencies encoded in ancestors. clusters is the list of all clusters
// including clusters that are not present in the ancestors map. has_successors
// is the set of clusters that are ancestors of some other cluster.
void TopologicalClusterSort(const std::unordered_set<string>& clusters,
                            const std::unordered_set<string>& has_successors,
                            const std::unordered_map<string, std::unordered_set<string>>& ancestors,
                            std::vector<string>* sorted)
{
    // The nodes are placed in 'sorted' in topological order.
    sorted->clear();
    // We don't use the standard DFS because we are not operating on Node*
    // objects.
    struct Work {
        string cluster;
        bool leave;
    };
    std::set<string> visited;
    std::vector<Work> stack;
    // Seed the processing list with clusters that have no successors.
    for (const auto& cluster : clusters) {
        if (has_successors.find(cluster) == has_successors.end()) {
            stack.push_back({cluster, false});
        }
    }
    while (!stack.empty()) {
        const Work item = stack.back();
        stack.pop_back();
        if (item.leave) {
            sorted->push_back(item.cluster);
            continue;
        }

        if (visited.find(item.cluster) != visited.end()) {
            continue;
        }
        visited.insert(item.cluster);

        stack.push_back({item.cluster, true});
        const auto& iter = ancestors.find(item.cluster);
        if (iter != ancestors.end()) {
            for (const auto& ancestor : iter->second) {
                stack.push_back({ancestor, false});
            }
        }
    }
    CHECK(sorted->size() == clusters.size());
}

}  // namespace

Node* Encapsulator::Subgraph::GetCallNode() const
{
    return call_node_;
}

int Encapsulator::Subgraph::GetArgIndexForEdge(const Edge* edge) const
{
    return args_by_dst_.at(InputTensor(edge->dst(), edge->dst_input()));
}

int Encapsulator::Subgraph::GetResultIndexForEdge(const Edge* edge) const
{
    return results_.at(OutputTensor(edge->src(), edge->src_output()));
}

Node* Encapsulator::Subgraph::GetRecvAtHostNode(const string& outside_compilation_subgraph_name) const
{
    return outside_compilation_subgraphs_.at(outside_compilation_subgraph_name).recv_at_host;
}

int Encapsulator::Subgraph::GetRecvAtHostSlot(const string& outside_compilation_subgraph_name, const Edge* edge) const
{
    return outside_compilation_subgraphs_.at(outside_compilation_subgraph_name)
        .inputs.at(OutputTensor(edge->src(), edge->src_output()));
}

Node* Encapsulator::Subgraph::GetSendFromHostNode(const string& outside_compilation_subgraph_name) const
{
    return outside_compilation_subgraphs_.at(outside_compilation_subgraph_name).send_from_host;
}

int Encapsulator::Subgraph::GetSendFromHostSlot(const string& outside_compilation_subgraph_name, const Edge* edge) const
{
    return outside_compilation_subgraphs_.at(outside_compilation_subgraph_name)
        .outputs_by_dst.at(InputTensor(edge->dst(), edge->dst_input()));
}

Node* Encapsulator::Subgraph::MakeNodeImage(const Graph* graph_in, Node* node)
{
    if (!graph_) {
        graph_.reset(new Graph(graph_in->op_registry()));
        graph_->set_versions(graph_in->versions());
    }

    // Enhance how the device for the encapsulated subgraph is
    // determined. In case of hard placement, ensure all the encapsulated nodes
    // have the same requested device, which in turn will be the requested device
    // for the entire encapsulated subgraph. In case of soft placement, use a
    // deterministic approach to fill in the requested device. Handle co-location
    // constraints similarly if they exist.
    if (device_.empty()) {
        device_ = node->assigned_device_name().empty() ? node->requested_device() : node->assigned_device_name();
    }

    return graph_->CopyNode(node);
}

Graph* Encapsulator::Subgraph::GetGraph() const
{
    return graph_.get();
}

Status Encapsulator::Subgraph::RecordArg(const Edge* edge, const std::unordered_map<const Node*, Node*>& node_images,
                                         std::vector<std::pair<const Node*, Node*>>* src_arg_pairs)
{
    Node* src_node = edge->src();
    int src_slot = edge->src_output();
    std::unordered_map<OutputTensor, int, OutputTensor::Hash>::iterator iter;
    bool inserted;
    std::tie(iter, inserted) = args_by_src_.emplace(OutputTensor(src_node, src_slot), args_by_src_.size());
    int arg_index = iter->second;
    if (inserted) {
        NodeDef arg_def;
        NodeDefBuilder builder(absl::StrCat(src_node->name(), "_", src_slot, "_arg"), kArgOp);
        DataType dtype = edge->dst()->input_type(edge->dst_input());
        builder.Attr("T", dtype);
        builder.Attr("index", arg_index);

        Status s = builder.Finalize(&arg_def);
        if (!s.ok())
            return s;

        Node* arg = graph_->AddNode(arg_def, &s);
        if (!s.ok())
            return s;

        src_arg_pairs->push_back({src_node, arg});
        args_.push_back(arg);
    }
    Node* dst_node = edge->dst();
    Node* dst_image = node_images.at(dst_node);
    int dst_slot = edge->dst_input();
    args_by_dst_[InputTensor(dst_node, dst_slot)] = arg_index;
    graph_->AddEdge(args_[arg_index], 0, dst_image, dst_slot);
    return absl::OkStatus();
}

Status Encapsulator::Subgraph::RecordResult(const Edge* edge, const std::unordered_map<const Node*, Node*>& node_images)
{
    Node* src_node = edge->src();
    Node* src_image = node_images.at(src_node);
    int src_slot = edge->src_output();
    std::unordered_map<OutputTensor, int, OutputTensor::Hash>::iterator iter;
    bool inserted;
    std::tie(iter, inserted) = results_.emplace(OutputTensor(src_node, src_slot), results_.size());
    int ret_index = iter->second;
    if (inserted) {
        NodeDef ret_def;
        NodeDefBuilder builder(absl::StrCat(src_node->name(), "_", src_slot, "_retval"), kRetValOp);
        DataType dtype = src_node->output_type(src_slot);
        builder.Attr("T", dtype);
        builder.Attr("index", ret_index);
        builder.Input(src_image->name(), src_slot, dtype);
        Status s = builder.Finalize(&ret_def);
        if (!s.ok())
            return s;
        Node* ret = graph_->AddNode(ret_def, &s);
        if (!s.ok())
            return s;

        graph_->AddEdge(src_image, src_slot, ret, 0);
    }
    return absl::OkStatus();
}

Encapsulator::Subgraph::OutsideCompilationSubgraph* Encapsulator::Subgraph::LookupOrCreateOutsideCompilationSubgraph(
    const string& outside_compilation_id)
{
    auto iter = outside_compilation_subgraphs_.emplace(outside_compilation_id, OutsideCompilationSubgraph()).first;
    OutsideCompilationSubgraph* outside_subgraph = &iter->second;
    return outside_subgraph;
}

void Encapsulator::Subgraph::RecordOutsideCompilationInputOrControl(const string& outside_compilation_id,
                                                                    const Edge* edge)
{
    OutsideCompilationSubgraph* outside_subgraph = LookupOrCreateOutsideCompilationSubgraph(outside_compilation_id);
    if (edge->IsControlEdge()) {
        outside_subgraph->control_inputs.insert(edge->src());
    } else {
        int input_index = outside_subgraph->inputs.size();
        outside_subgraph->inputs.emplace(OutputTensor(edge->src(), edge->src_output()), input_index);
    }
}

void Encapsulator::Subgraph::RecordOutsideCompilationOutputOrControl(const string& outside_compilation_id,
                                                                     const Edge* edge)
{
    OutsideCompilationSubgraph* outside_subgraph = LookupOrCreateOutsideCompilationSubgraph(outside_compilation_id);
    if (edge->IsControlEdge()) {
        outside_subgraph->control_outputs.insert(edge->dst());
    } else {
        DataType dtype = edge->dst()->input_type(edge->dst_input());
        auto output_iter =
            outside_subgraph->outputs_by_src
                .emplace(OutputTensor(edge->src(), edge->src_output()),
                         OutsideCompilationSubgraph::ArgNumAndType(outside_subgraph->outputs_by_src.size(), dtype))
                .first;
        const int output_index = output_iter->second.index;
        outside_subgraph->outputs_by_dst[InputTensor(edge->dst(), edge->dst_input())] = output_index;
    }
}

void Encapsulator::Subgraph::RecordOutsideCompilationDependency(const string& successor, const string& ancestor)
{
    outside_compilation_ancestors_[successor].insert(ancestor);
    outside_compilation_successors_[ancestor].insert(successor);
}

const std::unordered_map<string, std::unordered_set<string>> Encapsulator::Subgraph::OutsideCompilationAncestorMap()
    const
{
    return outside_compilation_ancestors_;
}

void Encapsulator::Subgraph::GetActiveClusterDependencyGraph(
    std::unordered_set<string>* clusters, std::unordered_set<string>* has_successor,
    std::unordered_map<string, std::unordered_set<string>>* ancestors_map)
{
    // During initial clustering the ancestor and successor datastructures may
    // have been built including oc_cluster names that never turned into subgraphs
    // because they had no edges into or out of the compiled cluster. Remove them
    // before proceeding to simplify the logic. Get the set of clusters that was
    // actually added, then remove references to the others.
    for (const auto& oc_subgraph : outside_compilation_subgraphs_) {
        clusters->insert(oc_subgraph.first);
    }
    for (const auto& cluster : outside_compilation_successors_) {
        if (clusters->find(cluster.first) != clusters->end()) {
            for (const auto& successor : cluster.second) {
                if (clusters->find(successor) != clusters->end()) {
                    has_successor->insert(cluster.first);
                    break;
                }
            }
        }
    }
    for (const auto& cluster : outside_compilation_ancestors_) {
        if (clusters->find(cluster.first) != clusters->end()) {
            std::unordered_set<string>& ancestors = (*ancestors_map)[cluster.first];
            for (const auto& ancestor : cluster.second) {
                if (clusters->find(ancestor) != clusters->end()) {
                    ancestors.insert(ancestor);
                }
            }
        }
    }
}

Status Encapsulator::Subgraph::AddHostComputes(const string& subgraph_name,
                                               const std::unordered_map<const Node*, Node*>& node_images)
{
    // Get the set of outside_compilation clusters and the dependency edges
    // between them.
    std::unordered_set<string> clusters;
    std::unordered_set<string> has_successor;
    std::unordered_map<string, std::unordered_set<string>> ancestors_map;
    GetActiveClusterDependencyGraph(&clusters, &has_successor, &ancestors_map);
    // Topologically sort the outside_compilation clusters according to their
    // dependency relation.
    std::vector<string> sorted_clusters;
    TopologicalClusterSort(clusters, has_successor, ancestors_map, &sorted_clusters);

    // The host compute nodes added for each outside_compilation_cluster;
    std::unordered_map<string, Node*> host_compute_node;
    for (const string& oc_subgraph_name : sorted_clusters) {
        OutsideCompilationSubgraph& oc_subgraph = outside_compilation_subgraphs_[oc_subgraph_name];
        if (!oc_subgraph.inputs.empty() || !oc_subgraph.control_inputs.empty() || !oc_subgraph.outputs_by_src.empty() ||
            !oc_subgraph.control_outputs.empty()) {
            // Build a _HostCompute node.
            std::vector<NodeDefBuilder::NodeOut> inputs(oc_subgraph.inputs.size());
            std::vector<DataType> input_dtypes(oc_subgraph.inputs.size(), DT_INVALID);
            std::vector<DataType> output_dtypes(oc_subgraph.outputs_by_src.size(), DT_INVALID);

            for (const auto& input_src : oc_subgraph.inputs) {
                const Node* src_node = input_src.first.node;
                Node* src_image = node_images.at(src_node);
                int src_slot = input_src.first.index;
                int input_index = input_src.second;

                DataType dtype = src_node->output_type(src_slot);
                inputs[input_index].Reset(src_image->name(), src_slot, dtype);
                input_dtypes[input_index] = dtype;
            }
            for (const auto& output : oc_subgraph.outputs_by_src) {
                DataType dtype = output.second.dtype;
                int output_index = output.second.index;
                output_dtypes[output_index] = dtype;
            }

            std::vector<string> host_compute_ancestors;
            const auto iter = ancestors_map.find(oc_subgraph_name);
            if (iter != ancestors_map.end()) {
                for (const string& ancestor_cluster : iter->second) {
                    host_compute_ancestors.push_back(
                        outside_compilation_subgraphs_[ancestor_cluster].host_compute_name);
                }
            }

            NodeDef host_compute_def;
            NodeDefBuilder builder(absl::StrCat("outside_compilation_", oc_subgraph_name, "_host_compute"),
                                   kHostComputeOp);
            builder.Input(inputs);
            builder.Attr("Tinputs", input_dtypes);
            builder.Attr("Toutputs", output_dtypes);
            builder.Attr("ancestors", host_compute_ancestors);
            builder.Attr("key", absl::StrCat("host_compute_channel_", subgraph_name, "_", oc_subgraph_name));
            builder.Attr("_outside_compilation_subgraph", oc_subgraph_name);
            Status s = builder.Finalize(&host_compute_def);
            if (!s.ok())
                return s;

            Node* host_compute = graph_->AddNode(host_compute_def, &s);
            if (!s.ok())
                return s;
            host_compute_node[host_compute->name()] = host_compute;
            oc_subgraph.host_compute_name = host_compute->name();

            // Connect the _HostCompute node to its producers in the subgraph.
            for (auto& input_src : oc_subgraph.inputs) {
                const Node* src_node = input_src.first.node;
                Node* src_image = node_images.at(src_node);
                int src_slot = input_src.first.index;
                int input_index = input_src.second;
                graph_->AddEdge(src_image, src_slot, host_compute, input_index);
            }

            // Connect the _HostCompute node to its control edge producers in the
            // subgraph.
            for (const auto& src_node : oc_subgraph.control_inputs) {
                Node* src_image = node_images.at(src_node);
                graph_->AddControlEdge(src_image, host_compute);
            }

            // Connect the _HostCompute node to its ancestor host compute nodes.
            for (const auto& ancestor_name : host_compute_ancestors) {
                Node* ancestor = host_compute_node[ancestor_name];
                graph_->AddControlEdge(ancestor, host_compute);
            }

            // Connect the consumers in the subgraph to the _HostCompute node.
            for (const auto& output : oc_subgraph.outputs_by_dst) {
                const Node* dst_node = output.first.node;
                Node* dst_image = node_images.at(dst_node);
                int dst_slot = output.first.index;
                int output_index = output.second;

                graph_->AddEdge(host_compute, output_index, dst_image, dst_slot);
            }

            // Connect the control edge consumers in the subgraph to the _HostCompute
            // node.
            for (const auto& dst_node : oc_subgraph.control_outputs) {
                Node* dst_image = node_images.at(dst_node);
                graph_->AddControlEdge(host_compute, dst_image);
            }
        }
    }

    return absl::OkStatus();
}

Status Encapsulator::Subgraph::MakeSequencingNode(const string& subgraph_name, Graph* graph_out)
{
    if (sequencer_ == nullptr) {
        NodeDef seq_def;
        NodeDefBuilder builder(absl::StrCat(subgraph_name, "_sequencer"), "NoOp");
        builder.Attr(kXlaHostTransferSequencerAttr, subgraph_name);
        builder.Device(device_);
        Status s = builder.Finalize(&seq_def);
        if (!s.ok())
            return s;

        sequencer_ = graph_out->AddNode(seq_def, &s);
        if (!s.ok())
            return s;
    }
    return absl::OkStatus();
}

void Encapsulator::Subgraph::ConnectSequencerToCallNode(Graph* graph_out)
{
    if (sequencer_ != nullptr) {
        VLOG(2) << "ConnectSequencerToCallNode";
        graph_out->AddControlEdge(sequencer_, call_node_);
    }
}

StatusOr<bool> Encapsulator::Subgraph::MaybeDefunctionalize(FunctionLibraryDefinition* library)
{
    bool is_defunc = false;
    for (int i = 2; i < graph_->num_node_ids(); ++i) {
        Node* node = graph_->FindNodeId(i);
        if (node == nullptr)
            continue;  // deleted node
        if (IsFunctionalControlFlowOps(node)) {
            TF_RETURN_IF_ERROR(DefunctionalizeFactory().defunctionalize(node, graph_.get(), *library));
            is_defunc = true;
        }
    }
    // NOTE(pengzhan): After lowering control flow ops, it is safe to leave device
    // attr empty, so we don't run Placer here.

    // TODO(pengzhan): Consider rerun the mark pass. This may bring some
    // performance gain.
    return is_defunc;
}

Status Encapsulator::Subgraph::AddShapeToFunctionDef(Graph& graph, FunctionDef* fdef)
{
    auto fdef_attr = fdef->mutable_attr();
    AttrValue attr_value;
    auto arg_shape = attr_value.mutable_func();
    *arg_shape->mutable_name() = "shape_info";
    for (auto& input_arg : fdef->signature().input_arg()) {
        for (auto n : graph.nodes()) {
            if (absl::AsciiStrToLower(n->name()) == input_arg.name() && n->attrs().Find("_output_shapes") != nullptr) {
                *(*arg_shape->mutable_attr())[input_arg.name()].mutable_shape() =
                    n->attrs().Find("_output_shapes")->shape();
            }
        }
    }
    (*fdef_attr)["input_args_info"] = attr_value;
    return absl::OkStatus();
}

Status Encapsulator::Subgraph::BuildFunctionDef(const string& name_in, const RewriteSubgraphFn& rewrite_subgraph_fn,
                                                bool reuse_existing_functions, FunctionLibraryDefinition* library)
{
    // name_in is copied here because name may be modified below if
    // rewrite_subgraph_fn is true.
    string name = name_in;
    call_node_def_.set_op(name);
    call_node_def_.set_name(name);
    call_node_def_.set_device(device_);

    if (rewrite_subgraph_fn) {
        std::vector<OutputTensor> arg_source_tensors(args_by_src_.size());
        for (const auto& arg : args_by_src_) {
            arg_source_tensors.at(arg.second) = arg.first;
        }
        // Initialize the input and output permutations to the identity.
        std::vector<int> input_permutation(args_by_src_.size());
        std::iota(input_permutation.begin(), input_permutation.end(), 0);
        std::vector<int> output_permutation(results_.size());
        std::iota(output_permutation.begin(), output_permutation.end(), 0);

        TF_RETURN_IF_ERROR(
            rewrite_subgraph_fn(arg_source_tensors, &graph_, &input_permutation, &output_permutation, &call_node_def_));

        // Apply the input/output permutations to the 'args_by_...' and 'results_'
        // mappings, so when we build edges in BuildOutputGraph() we
        // connect them to the right input/output positions.
        if (input_permutation.size() != args_by_src_.size()) {
            return errors::InvalidArgument("Input permutation has incorrect size.");
        }
        if (output_permutation.size() != results_.size()) {
            return errors::InvalidArgument("Output permutation has incorrect size.");
        }
        for (auto& arg : args_by_src_) {
            arg.second = input_permutation[arg.second];
        }
        for (auto& arg : args_by_dst_) {
            arg.second = input_permutation[arg.second];
        }
        for (auto& result : results_) {
            result.second = output_permutation[result.second];
        }

        name = call_node_def_.op();
    }

    function_def_name_ = name;

    FunctionDef fdef;
    TF_RETURN_IF_ERROR(GraphToFunctionDef(*graph_, name, &fdef));
    TF_RETURN_IF_ERROR(AddShapeToFunctionDef(*graph_, &fdef));

    if (VLOG_IS_ON(1)) {
        VLOG(2) << "Build function def " << name;
        dump_graph::DumpGraphToFile(absl::StrCat("encapsulate_fdef_graph_", name), *graph_, library);
        dump_graph::DumpFunctionDefToFile(absl::StrCat("encapsulate_fdef_", name), fdef);
    }

    if (!reuse_existing_functions || library->Find(name) == nullptr) {
        TF_RETURN_IF_ERROR(library->AddFunctionDef(fdef));
    }

    string defunct_name = name + kDefunctionalizedSuffix;
    if (library->Find(defunct_name) == nullptr) {
        TF_ASSIGN_OR_RETURN(bool is_defunc, MaybeDefunctionalize(library));
        if (is_defunc) {
            FunctionDef fdef_defunc;
            TF_RETURN_IF_ERROR(GraphToFunctionDef(*graph_, defunct_name, &fdef_defunc));
            TF_RETURN_IF_ERROR(library->AddFunctionDef(fdef_defunc));

            VLOG(2) << dump_graph::DumpFunctionDefToFile(absl::StrCat("encapsulate_fdef_defunc_", defunct_name),
                                                         fdef_defunc);
        }
    }

    return absl::OkStatus();
}

Status Encapsulator::Subgraph::AddShapeInferenceInfo(const string& subgraph_name,
                                                     const string& outside_compilation_subgraph_name,
                                                     const std::vector<TensorShapeProto>& shapes,
                                                     Graph* inference_graph, FunctionLibraryDefinition* library)
{
    OutsideCompilationSubgraph& oc_subgraph = outside_compilation_subgraphs_.at(outside_compilation_subgraph_name);

    Node* host_compute = nullptr;
    for (Node* n : graph_->nodes()) {
        if (n->name() == oc_subgraph.host_compute_name) {
            host_compute = n;
            break;
        }
    }
    if (host_compute == nullptr) {
        return errors::InvalidArgument("After rewriting subgraph ", outside_compilation_subgraph_name,
                                       " there is no HostCompute Op for outside compilation subgraph ",
                                       oc_subgraph.host_compute_name);
    }

    if (inference_graph == nullptr) {
        host_compute->AddAttr("shape_inference_graph", "");
        host_compute->AddAttr("shapes", shapes);
    } else {
        string inference_graph_name = absl::StrCat("_outside_compilation_shape_inference_", subgraph_name, "_",
                                                   outside_compilation_subgraph_name);
        FunctionDef fdef;
        TF_RETURN_IF_ERROR(GraphToFunctionDef(*inference_graph, inference_graph_name, &fdef));
        host_compute->AddAttr("shape_inference_graph", inference_graph_name);
        host_compute->AddAttr("shapes", std::vector<TensorShapeProto>());
        // TODO(sibyl-Aix6ihai): Understand why there are multiple calls to
        // Encapsulator.
        if (library->Find(inference_graph_name) == nullptr) {
            TF_RETURN_IF_ERROR(library->AddFunctionDef(fdef));
        }
    }
    return absl::OkStatus();
}

Status Encapsulator::Subgraph::ReplaceFunctionDef(FunctionLibraryDefinition* library)
{
    const string& name = function_def_name_;

    FunctionDef fdef;
    TF_RETURN_IF_ERROR(GraphToFunctionDef(*graph_, name, &fdef));

    if (VLOG_IS_ON(1)) {
        VLOG(2) << "Replace function def " << name;
        dump_graph::DumpGraphToFile(absl::StrCat("replace_encapsulate_fdef_graph_", name), *graph_, library);
        dump_graph::DumpFunctionDefToFile(absl::StrCat("replace_encapsulate_fdef_", name), fdef);
    }

    TF_RETURN_IF_ERROR(library->ReplaceFunction(name, fdef));
    return absl::OkStatus();
}

Status Encapsulator::Subgraph::AddFunctionCallNode(const std::unordered_map<const Node*, Node*>& node_images,
                                                   Graph* graph_out)
{
    Status s;
    call_node_ = graph_out->AddNode(call_node_def_, &s);
    if (!s.ok())
        return s;

    // Copy the assigned device and the key_annotation over.
    call_node_->set_assigned_device_name(device_);

    return absl::OkStatus();
}

Status Encapsulator::Subgraph::AddHostComputeKeyPlaceholder(OutsideCompilationSubgraph* oc_subgraph, Graph* graph_out)
{
    TensorShapeProto shape_proto;
    TensorShape shape({2});
    shape.AsProto(&shape_proto);
    GraphDefBuilder::Options options(graph_out, /*status=*/nullptr);
    NodeDef key_def;
    NodeDefBuilder builder(absl::StrCat(call_node_def_.name(), "_key_placeholder"), "Placeholder");
    builder.Attr("dtype", DT_STRING);
    builder.Attr("shape", shape_proto);
    builder.Attr("_host_compute_call_node", call_node_def_.name());
    Status s = builder.Finalize(&key_def);
    if (!s.ok())
        return s;

    host_compute_key_placeholder_ = graph_out->AddNode(key_def, &s);
    if (!s.ok())
        return s;
    host_compute_key_placeholder_->set_assigned_device_name(device_);

    return absl::OkStatus();
}

Status Encapsulator::Subgraph::AddRecvAtHostNode(const string& group_attribute, const string& subgraph_name,
                                                 const string& outside_compilation_attribute,
                                                 const string& oc_subgraph_name,
                                                 OutsideCompilationSubgraph* oc_subgraph, Graph* graph_out)
{
    if (host_compute_key_placeholder_ == nullptr) {
        TF_RETURN_IF_ERROR(AddHostComputeKeyPlaceholder(oc_subgraph, graph_out));
    }

    std::vector<DataType> dtypes(oc_subgraph->inputs.size(), DT_INVALID);

    for (const auto& input : oc_subgraph->inputs) {
        const Node* src_node = input.first.node;
        int src_slot = input.first.index;
        int input_index = input.second;

        DataType dtype = src_node->output_type(src_slot);
        dtypes[input_index] = dtype;
    }

    NodeDef recv_def;
    NodeDefBuilder builder(absl::StrCat("outside_compilation_", subgraph_name, "_", oc_subgraph_name, "_recv"),
                           kRecvAtHostOp);
    builder.Device(device_);
    builder.Attr("Toutputs", dtypes);
    // The correct device_ordinal will be inserted during replication in a
    // subsequent rewrite.
    builder.Attr("device_ordinal", 0);
    builder.Attr("key", absl::StrCat("host_compute_channel_", subgraph_name, "_", oc_subgraph_name));
    builder.Attr(group_attribute, subgraph_name);
    builder.Attr(outside_compilation_attribute, oc_subgraph_name);
    builder.Input(host_compute_key_placeholder_->name(), 0, DT_STRING);
    Status s = builder.Finalize(&recv_def);
    if (!s.ok())
        return s;

    oc_subgraph->recv_at_host = graph_out->AddNode(recv_def, &s);
    if (!s.ok())
        return s;
    graph_out->AddEdge(host_compute_key_placeholder_, 0, oc_subgraph->recv_at_host, 0);

    // Add a control dependency forcing the RecvAtHost to run before the subgraph
    // completes. This has no effect on execution order but prevents the
    // RecvAtHost being pruned.
    TF_RETURN_IF_ERROR(MakeSequencingNode(subgraph_name, graph_out));
    graph_out->AddControlEdge(oc_subgraph->recv_at_host, sequencer_);

    return absl::OkStatus();
}

Status Encapsulator::Subgraph::AddSendFromHostNode(const std::unordered_map<const Node*, Node*>& node_images,
                                                   const string& group_attribute, const string& subgraph_name,
                                                   const string& outside_compilation_attribute,
                                                   const string& oc_subgraph_name,
                                                   OutsideCompilationSubgraph* oc_subgraph, Graph* graph_out)
{
    if (host_compute_key_placeholder_ == nullptr) {
        TF_RETURN_IF_ERROR(AddHostComputeKeyPlaceholder(oc_subgraph, graph_out));
    }

    std::vector<DataType> dtypes(oc_subgraph->outputs_by_src.size(), DT_INVALID);
    std::vector<NodeDefBuilder::NodeOut> inputs(oc_subgraph->outputs_by_src.size());

    for (const auto& output : oc_subgraph->outputs_by_src) {
        const Node* src_node = output.first.node;
        Node* src_image = node_images.at(src_node);
        int src_slot = output.first.index;
        int output_index = output.second.index;

        DataType dtype = src_node->output_type(src_slot);
        dtypes[output_index] = dtype;
        inputs[output_index].Reset(src_image->name(), src_slot, dtype);
    }

    NodeDef send_def;
    NodeDefBuilder builder(absl::StrCat("outside_compilation_", subgraph_name, "_", oc_subgraph_name, "_send"),
                           kSendFromHostOp);
    builder.Device(device_);
    builder.Attr("Tinputs", dtypes);
    builder.Attr("key", absl::StrCat("host_compute_channel_", subgraph_name, "_", oc_subgraph_name));
    // The correct device_ordinal will be inserted during replication in a
    // subsequent rewrite.
    builder.Attr("device_ordinal", 0);
    builder.Attr(group_attribute, subgraph_name);
    builder.Attr(outside_compilation_attribute, oc_subgraph_name);
    builder.Input(inputs);
    builder.Input(host_compute_key_placeholder_->name(), 0, DT_STRING);
    Status s = builder.Finalize(&send_def);
    if (!s.ok())
        return s;

    oc_subgraph->send_from_host = graph_out->AddNode(send_def, &s);
    if (!s.ok())
        return s;
    graph_out->AddEdge(host_compute_key_placeholder_, 0, oc_subgraph->send_from_host, inputs.size());

    // Add a control dependency forcing the SendFromHost to run before the
    // subgraph completes. This has no effect on execution order but prevents the
    // RecvAtHost being pruned.
    TF_RETURN_IF_ERROR(MakeSequencingNode(subgraph_name, graph_out));
    graph_out->AddControlEdge(oc_subgraph->send_from_host, sequencer_);

    return absl::OkStatus();
}

Status Encapsulator::Subgraph::AddOutsideCompilationHostIONodes(
    const string& group_attribute, const string& subgraph_name, const string& outside_compilation_attribute,
    const std::unordered_map<const Node*, Node*>& node_images, Graph* graph_out)
{
    for (auto& outside_compilation_subgraph_entry : outside_compilation_subgraphs_) {
        const string& oc_name = outside_compilation_subgraph_entry.first;
        OutsideCompilationSubgraph& oc_subgraph = outside_compilation_subgraph_entry.second;

        if (!oc_subgraph.inputs.empty() || !oc_subgraph.control_inputs.empty()) {
            TF_RETURN_IF_ERROR(AddRecvAtHostNode(group_attribute, subgraph_name, outside_compilation_attribute, oc_name,
                                                 &oc_subgraph, graph_out));
        }

        if (!oc_subgraph.outputs_by_src.empty() || !oc_subgraph.control_outputs.empty()) {
            TF_RETURN_IF_ERROR(AddSendFromHostNode(node_images, group_attribute, subgraph_name,
                                                   outside_compilation_attribute, oc_name, &oc_subgraph, graph_out));
        }
    }
    return absl::OkStatus();
}

void Encapsulator::Subgraph::GetOutsideCompilationSubgraphNames(std::vector<string>* names) const
{
    for (auto& entry : outside_compilation_subgraphs_) {
        names->push_back(entry.first);
    }
}

Status Encapsulator::GetFunctionNameAttr(Node const* node, string* attr, string* outside_compilation_attr) const
{
    AttrSlice attrs = node->attrs();
    attr->clear();
    outside_compilation_attr->clear();
    bool found_group_attribute = false;
    bool found_outside_compilation_attribute = false;
    for (const auto& node_attr : attrs) {
        if (node_attr.first == group_attribute_) {
            TF_RETURN_IF_ERROR(AttrValueHasType(node_attr.second, "string"));
            *attr = node_attr.second.s();
            found_group_attribute = true;
        } else if (node_attr.first == outside_compilation_attribute_) {
            TF_RETURN_IF_ERROR(AttrValueHasType(node_attr.second, "string"));
            *outside_compilation_attr = node_attr.second.s();
            found_outside_compilation_attribute = true;
        }
        if (found_group_attribute && found_outside_compilation_attribute)
            break;
    }

    if (found_outside_compilation_attribute && !found_group_attribute) {
        return errors::InvalidArgument("Node ", node->name(), " has ", outside_compilation_attribute_,
                                       " attribute but no ", group_attribute_, " attribute.");
    } else {
        return absl::OkStatus();
    }
}

bool IsInSubgraph(const string& func_id, const string& outside_compilation_id)
{
    return !func_id.empty() && outside_compilation_id.empty();
}

Status Encapsulator::CopySubgraphNodes(std::unordered_map<const Node*, Node*>* node_images)
{
    for (Node* node : graph_in_->op_nodes()) {
        string func_id;
        string outside_compilation_id;
        TF_RETURN_IF_ERROR(GetFunctionNameAttr(node, &func_id, &outside_compilation_id));
        if (!IsInSubgraph(func_id, outside_compilation_id))
            continue;

        Subgraph& subgraph = subgraphs_[func_id];
        Node* image = subgraph.MakeNodeImage(graph_in_, node);
        image->ClearAttr(group_attribute_);
        image->ClearAttr("_grappler:ArithmeticOptimizer:MinimizeBroadcasts");
        image->ClearAttr("_grappler:ArithmeticOptimizer:AddOpsRewriteStage");
        // `tao_compiler_main` failed to load graphdef with ops having
        // `_output_shapes`. Just remove such attribute here as a workaround to fix
        // such problem.
        image->ClearAttr("_output_shapes");
        (*node_images)[node] = image;
    }
    return absl::OkStatus();
}

Status Encapsulator::CopySubgraphEdges(const std::unordered_map<const Node*, Node*>& node_images,
                                       std::vector<std::pair<const Node*, Node*>>* src_arg_pairs)
{
    for (const Edge* edge : graph_in_->edges()) {
        string src_func_id;
        string src_outside_compilation_id;
        TF_RETURN_IF_ERROR(GetFunctionNameAttr(edge->src(), &src_func_id, &src_outside_compilation_id));
        string dst_func_id;
        string dst_outside_compilation_id;
        TF_RETURN_IF_ERROR(GetFunctionNameAttr(edge->dst(), &dst_func_id, &dst_outside_compilation_id));
        Node* src_image = gtl::FindWithDefault(node_images, edge->src(), nullptr);
        Node* dst_image = gtl::FindWithDefault(node_images, edge->dst(), nullptr);

        // Copy edges that are local to a subgraph.
        if (IsInSubgraph(src_func_id, src_outside_compilation_id) &&
            IsInSubgraph(dst_func_id, dst_outside_compilation_id) && src_func_id == dst_func_id) {
            Graph* g = subgraphs_[src_func_id].GetGraph();
            if (edge->IsControlEdge()) {
                g->AddControlEdge(src_image, dst_image);
            } else {
                g->AddEdge(src_image, edge->src_output(), dst_image, edge->dst_input());
            }
            continue;
        }

        // Record 'src' as an output of its subgraph, if applicable.
        if (IsInSubgraph(src_func_id, src_outside_compilation_id)) {
            if (!edge->IsControlEdge()) {
                DataType dtype = edge->src()->output_type(edge->src_output());
                if (IsRefType(dtype)) {
                    return errors::InvalidArgument("Ref Tensors (e.g., Variables) are not supported as results: "
                                                   "tensor ",
                                                   edge->src()->name(), ":", edge->src_output());
                }
            }

            Subgraph& src_subgraph = subgraphs_[src_func_id];
            if (src_func_id == dst_func_id) {
                // src is in the subgraph and dst is outside_compilation in the same
                // subgraph.
                src_subgraph.RecordOutsideCompilationInputOrControl(dst_outside_compilation_id, edge);
            } else {
                // Ignore control edges leaving the subgraph. We will lift them onto the
                // enclosing call operators in BuildOutputGraph().
                if (!edge->IsControlEdge()) {
                    TF_RETURN_IF_ERROR(src_subgraph.RecordResult(edge, node_images));
                }
            }
        }

        // Record 'dst' as an input of its subgraph, if applicable.
        if (IsInSubgraph(dst_func_id, dst_outside_compilation_id)) {
            // Look at the type of the destination not the source, since Ref output
            // Tensors can be automatically cast to non-Ref Tensors at the
            // destination.
            if (!edge->IsControlEdge()) {
                DataType dtype = edge->dst()->input_type(edge->dst_input());
                if (IsRefType(dtype)) {
                    return errors::InvalidArgument("Ref Tensors (e.g., Variables) are not supported as args: "
                                                   "tensor ",
                                                   edge->src()->name(), ":", edge->src_output());
                }
            }

            Subgraph& dst_subgraph = subgraphs_[dst_func_id];
            if (src_func_id == dst_func_id) {
                // dst is in the subgraph and src is outside_compilation in the same
                // subgraph.
                dst_subgraph.RecordOutsideCompilationOutputOrControl(src_outside_compilation_id, edge);
            } else {
                // Ignore control edges entering the subgraph. We will lift them onto
                // the enclosing call operators in BuildOutputGraph().
                if (!edge->IsControlEdge()) {
                    TF_RETURN_IF_ERROR(dst_subgraph.RecordArg(edge, node_images, src_arg_pairs));
                }
            }
        }
    }
    return absl::OkStatus();
}

Status Encapsulator::SplitIntoSubgraphs(FunctionLibraryDefinition* library)
{
    Status s;

    // Map from input graph nodes to subgraph nodes.
    std::unordered_map<const Node*, Node*> node_images;

    // Each entry of src_arg_pairs is a pair whose first element is a node in the
    // original graph that has an output edge in the subgraph, and whose second
    // element is the arg node in the subgraph that it sends to. The vector will
    // be filled in below in AddArgs.
    std::vector<std::pair<const Node*, Node*>> src_arg_pairs;

    TF_RETURN_IF_ERROR(CopySubgraphNodes(&node_images));
    TF_RETURN_IF_ERROR(CopySubgraphEdges(node_images, &src_arg_pairs));

    // For each subgraph, add the nodes that deal with inputs and outputs its
    // nested outside_compilation subgraphs. These could not be added earlier
    // during CopySubgraphEdges since we need to discover all the types of the
    // inputs and outputs for an outside_compilation subgraph before creating a
    // single input and output node for it.
    for (auto& entry : subgraphs_) {
        Subgraph& subgraph = entry.second;
        TF_RETURN_IF_ERROR(subgraph.AddHostComputes(entry.first, node_images));
    }

    MarkGuaranteedConstants(*graph_in_, src_arg_pairs);

    for (auto& entry : subgraphs_) {
        Subgraph& subgraph = entry.second;
        FixupSourceAndSinkEdges(subgraph.GetGraph());
        // Verify that the graph has well-formed control flow structure.
        std::vector<ControlFlowInfo> dummy;
        TF_RETURN_IF_ERROR(BuildControlFlowInfo(subgraph.GetGraph(), &dummy));
    }

    if (VLOG_IS_ON(1)) {
        // Dump subgraphs.
        for (auto& entry : subgraphs_) {
            dump_graph::DumpGraphToFile(absl::StrCat("encapsulate_subgraphs_subgraph_", entry.first),
                                        *entry.second.GetGraph(), library);
        }
    }

    return s;
}

Status Encapsulator::BuildFunctionDefs(const RewriteSubgraphFn& rewrite_subgraph_fn, bool reuse_existing_functions,
                                       FunctionLibraryDefinition* library)
{
    for (auto& subgraph_entry : subgraphs_) {
        string name = subgraph_entry.first;
        Subgraph& subgraph = subgraph_entry.second;
        TF_RETURN_IF_ERROR(subgraph.BuildFunctionDef(name, rewrite_subgraph_fn, reuse_existing_functions, library));
    }
    return absl::OkStatus();
}

Status Encapsulator::CopyNodesToOutputGraph(Graph* graph_out, std::unordered_map<const Node*, Node*>* node_images)
{
    for (Node* node : graph_in_->op_nodes()) {
        string func_id;
        string outside_compilation_id;
        TF_RETURN_IF_ERROR(GetFunctionNameAttr(node, &func_id, &outside_compilation_id));

        // Don't copy nodes that are going to be encapsulated.
        if (IsInSubgraph(func_id, outside_compilation_id))
            continue;

        Node* image = graph_out->CopyNode(node);
        (*node_images)[node] = image;
    }
    (*node_images)[graph_in_->source_node()] = graph_out->source_node();
    (*node_images)[graph_in_->sink_node()] = graph_out->sink_node();
    return absl::OkStatus();
}

Status Encapsulator::AddFunctionCallNodes(const std::unordered_map<const Node*, Node*>& node_images, Graph* graph_out)
{
    for (auto& subgraph_entry : subgraphs_) {
        TF_RETURN_IF_ERROR(subgraph_entry.second.AddFunctionCallNode(node_images, graph_out));
    }
    return absl::OkStatus();
}

Status Encapsulator::AddOutsideCompilationHostIONodes(const std::unordered_map<const Node*, Node*>& node_images,
                                                      Graph* graph_out)
{
    for (auto& subgraph_entry : subgraphs_) {
        const string& subgraph_name = subgraph_entry.first;
        Subgraph& subgraph = subgraph_entry.second;
        TF_RETURN_IF_ERROR(subgraph.AddOutsideCompilationHostIONodes(
            group_attribute_, subgraph_name, outside_compilation_attribute_, node_images, graph_out));
    }
    return absl::OkStatus();
}

Status Encapsulator::FindOutputImageOfEdgeSrc(const string& src_func_id, const string& src_outside_compilation_id,
                                              const string& dst_func_id, const string& dst_outside_compilation_id,
                                              const std::unordered_map<const Node*, Node*>& node_images,
                                              const Node* original_src_node, Node** src_image)
{
    if (IsInSubgraph(src_func_id, src_outside_compilation_id)) {
        if (dst_func_id == src_func_id) {
            // The edge is from a subgraph to an outside_compilation cluster in the
            // same subgraph so use the appropriate _RecvAtHost node in the output
            // graph.
            TF_RET_CHECK(!dst_outside_compilation_id.empty());
            *src_image = subgraphs_.at(src_func_id).GetRecvAtHostNode(dst_outside_compilation_id);
        } else {
            // The edge is from a subgraph to a regular node in the output graph so
            // use the subgraph's call node output.
            *src_image = subgraphs_.at(src_func_id).GetCallNode();
        }
    } else {
        // The source of the edge is in the output graph so use the node image in
        // the output graph.
        *src_image = node_images.at(original_src_node);
    }
    return absl::OkStatus();
}

int Encapsulator::FindOutputSlotOfEdgeSrc(const string& src_func_id, const string& src_outside_compilation_id,
                                          const string& dst_func_id, const string& dst_outside_compilation_id,
                                          const Edge* edge)
{
    if (IsInSubgraph(src_func_id, src_outside_compilation_id)) {
        const Subgraph& src_subgraph = subgraphs_.at(src_func_id);
        if (src_func_id == dst_func_id) {
            // 'src' is in a subgraph and 'dst' is outside_compilation in the same
            // subgraph. Use the corresponding _RecvAtHost output instead.
            return src_subgraph.GetRecvAtHostSlot(dst_outside_compilation_id, edge);
        } else {
            // 'src' is in a subgraph and 'dst' is a regular node in the output
            // graph. Use the corresponding call output instead.
            return src_subgraph.GetResultIndexForEdge(edge);
        }
    } else {
        // The source of the edge is in the output graph so use the regular edge
        // slot.
        return edge->src_output();
    }
}

Status Encapsulator::FindOutputImageOfEdgeDst(const string& src_func_id, const string& src_outside_compilation_id,
                                              const string& dst_func_id, const string& dst_outside_compilation_id,
                                              const std::unordered_map<const Node*, Node*>& node_images,
                                              const Node* original_dst_node, Node** dst_image)
{
    if (IsInSubgraph(dst_func_id, dst_outside_compilation_id)) {
        if (src_func_id == dst_func_id) {
            // The edge is to a subgraph from an outside_compilation cluster in the
            // same subgraph so use the appropriate _SendFromHost node in the output
            // graph.
            TF_RET_CHECK(!src_outside_compilation_id.empty());
            *dst_image = subgraphs_.at(dst_func_id).GetSendFromHostNode(src_outside_compilation_id);
        } else {
            // The edge is to a subgraph from a regular node in the output graph so
            // use the subgraph's call node input.
            *dst_image = subgraphs_.at(dst_func_id).GetCallNode();
        }
    } else {
        // The destination of the edge is in the output graph so use the node image
        // in the output graph.
        *dst_image = node_images.at(original_dst_node);
    }
    return absl::OkStatus();
}

int Encapsulator::FindOutputSlotOfEdgeDst(const string& src_func_id, const string& src_outside_compilation_id,
                                          const string& dst_func_id, const string& dst_outside_compilation_id,
                                          const Edge* edge)
{
    if (IsInSubgraph(dst_func_id, dst_outside_compilation_id)) {
        const Subgraph& dst_subgraph = subgraphs_.at(dst_func_id);
        if (dst_func_id == src_func_id) {
            // 'dst' is in a subgraph and 'src' is outside_compilation in the same
            // subgraph. Use the corresponding _SendFromHost input instead.
            return dst_subgraph.GetSendFromHostSlot(src_outside_compilation_id, edge);
        } else {
            // 'dst' is in a subgraph and 'src' is a regular node in the output
            // graph. Use the corresponding call input instead.
            return dst_subgraph.GetArgIndexForEdge(edge);
        }
    } else {
        // The destination of the edge is in the output graph so use the regular
        // edge slot.
        return edge->dst_input();
    }
}

Status Encapsulator::CopyEdgeToOutputGraph(
    const Edge* edge, const string& src_func_id, const string& src_outside_compilation_id, const string& dst_func_id,
    const string& dst_outside_compilation_id, const std::unordered_map<const Node*, Node*>& node_images,
    Graph* graph_out,
    std::unordered_set<std::pair<OutputTensor, InputTensor>, OutputInputTensorPairHasher>* edges_added)
{
    Node* src_image = nullptr;
    TF_RETURN_IF_ERROR(FindOutputImageOfEdgeSrc(src_func_id, src_outside_compilation_id, dst_func_id,
                                                dst_outside_compilation_id, node_images, edge->src(), &src_image));
    Node* dst_image = nullptr;
    TF_RETURN_IF_ERROR(FindOutputImageOfEdgeDst(src_func_id, src_outside_compilation_id, dst_func_id,
                                                dst_outside_compilation_id, node_images, edge->dst(), &dst_image));

    // If this is a control edge then copy it and return. Lift control edges onto
    // the enclosing call operator.
    if (edge->IsControlEdge()) {
        // Add the control edge, if we have not already added it, using the images
        // determined above (potentially call operators or RecvAtHost/SendFromHost).
        if (edges_added->emplace(OutputTensor(src_image, -1), InputTensor(dst_image, -1)).second) {
            graph_out->AddControlEdge(src_image, dst_image);
        }

        return absl::OkStatus();
    }

    int src_output =
        FindOutputSlotOfEdgeSrc(src_func_id, src_outside_compilation_id, dst_func_id, dst_outside_compilation_id, edge);

    int dst_input =
        FindOutputSlotOfEdgeDst(src_func_id, src_outside_compilation_id, dst_func_id, dst_outside_compilation_id, edge);

    // Add the edge, if we have not already added it.
    if (edges_added->emplace(OutputTensor(src_image, src_output), InputTensor(dst_image, dst_input)).second) {
        graph_out->AddEdge(src_image, src_output, dst_image, dst_input);
    }
    return absl::OkStatus();
}

Status Encapsulator::AddCallNodeDependencies(Graph* graph_out)
{
    for (const auto& ancestors : subgraph_ancestors_) {
        const string& subgraph = ancestors.first;
        for (const string& ancestor : ancestors.second) {
            graph_out->AddControlEdge(subgraphs_[ancestor].GetCallNode(), subgraphs_[subgraph].GetCallNode());
        }
    }
    return absl::OkStatus();
}

Status Encapsulator::AddEdgesToOutputGraph(const std::unordered_map<const Node*, Node*>& node_images, Graph* graph_out)
{
    // Set of edges already added to the output graph, represented as (src, dst)
    // pairs. We use the set to deduplicate edges; multiple edges in the input
    // graph may map to one edge in the output graph.
    std::unordered_set<std::pair<OutputTensor, InputTensor>, OutputInputTensorPairHasher> edges_added;

    for (const Edge* edge : graph_in_->edges()) {
        string src_func_id;
        string src_outside_compilation_id;
        TF_RETURN_IF_ERROR(GetFunctionNameAttr(edge->src(), &src_func_id, &src_outside_compilation_id));
        string dst_func_id;
        string dst_outside_compilation_id;
        TF_RETURN_IF_ERROR(GetFunctionNameAttr(edge->dst(), &dst_func_id, &dst_outside_compilation_id));

        // Ignore edges that are strictly contained within one subgraph, unless
        // we are constructing parallel check graphs.
        if (IsInSubgraph(src_func_id, src_outside_compilation_id) &&
            IsInSubgraph(dst_func_id, dst_outside_compilation_id) && src_func_id == dst_func_id) {
            continue;
        }

        // We have an edge that crosses a cluster boundary or is entirely within the
        // unclustered graph.
        TF_RETURN_IF_ERROR(CopyEdgeToOutputGraph(edge, src_func_id, src_outside_compilation_id, dst_func_id,
                                                 dst_outside_compilation_id, node_images, graph_out, &edges_added));
    }

    for (auto& subgraph_entry : subgraphs_) {
        Subgraph& subgraph = subgraph_entry.second;
        subgraph.ConnectSequencerToCallNode(graph_out);
    }
    TF_RETURN_IF_ERROR(AddCallNodeDependencies(graph_out));

    return absl::OkStatus();
}

}  // namespace

}  // namespace npu_xla
}  // namespace tensorflow