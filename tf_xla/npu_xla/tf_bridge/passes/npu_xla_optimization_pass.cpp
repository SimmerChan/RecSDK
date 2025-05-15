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

#include "tf_bridge/passes/npu_xla_optimization_pass.h"

#include "tf_bridge/passes/build_npu_xla_op_pass.h"
#include "tf_bridge/passes/encapsulate_subgraphs_pass.h"
#include "tf_bridge/passes/mark_for_npu_compilation_pass.h"
#include "tf_bridge/tf/log.h"
#include "tf_bridge/utils/dump_graph.h"

namespace tensorflow {
namespace npu_xla {

namespace {
void DumpGraph(const GraphOptimizationPassOptions& options, const char* const name)
{
    if (VLOG_IS_ON(VLOG_LEVEL_2)) {
        Graph* graph = options.graph->get();
        auto dumped = dump_graph::DumpGraphToFile(name, *graph, options.flib_def);
        VLOG(VLOG_LEVEL_2) << "NpuXlaOptimizationPass dump graph: " << dumped;
    }
}

}  // namespace

NpuXlaOptimizationPass::NpuXlaOptimizationPass() {}
Status NpuXlaOptimizationPass::Run(const GraphOptimizationPassOptions& options)
{
    DumpGraph(options, "before_npu_xla_optimization_pass");

    MarkForNpuCompilationPass mark_pass;
    mark_pass.set_name("MarkForNpuCompilationPass");
    TF_RETURN_IF_ERROR(mark_pass.Run(options));

    EncapsulateSubgraphsPass encap_pass;
    encap_pass.set_name("EncapsulateSubgraphsPass");
    TF_RETURN_IF_ERROR(encap_pass.Run(options));

    DumpGraph(options, "before_build_npu_xla_op_pass");

    BuildNpuXlaOpPass op_pass;
    op_pass.set_name("BuildNpuXlaOpPass");
    TF_RETURN_IF_ERROR(op_pass.Run(options));

    DumpGraph(options, "after_npu_xla_optimization_pass");

    return OkStatus();
}

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 0, NpuXlaOptimizationPass);
}  // namespace npu_xla
}  // namespace tensorflow