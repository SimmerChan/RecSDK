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

#ifndef NPU_XLA_TF_BRIDGE_UTILS_DUMP_GRAPH_H_
#define NPU_XLA_TF_BRIDGE_UTILS_DUMP_GRAPH_H_

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {
namespace npu_xla {
namespace dump_graph {
string DumpGraphDefToFile(const string& name, GraphDef const& graph_def);

string DumpGraphToFile(const string& name, Graph const& graph, const FunctionLibraryDefinition* flib_def = nullptr);

// Similar to DumpGraphDefToFile, but dumps a function as a FunctionDef text
// proto. Returns the file name chosen.
string DumpFunctionDefToFile(const string& name, FunctionDef const& fdef);
}  // namespace dump_graph
}  // namespace npu_xla
}  // namespace tensorflow

#endif  // NPU_XLA_TF_BRIDGE_UTILS_DUMP_GRAPH_H_
