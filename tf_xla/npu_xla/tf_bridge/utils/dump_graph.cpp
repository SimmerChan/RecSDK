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

#include "tf_bridge/utils/dump_graph.h"

#include <unordered_map>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/platform/env.h"
#include "tf_bridge/utils/dump_options.h"

namespace tensorflow {
namespace npu_xla {
namespace dump_graph {

#define VLOG_LEVEL_2 VLOG(2)

namespace {

string WriteTextProtoToUniqueFile(Env* env, const string& name, const char* proto_type,
                                  const ::tensorflow::protobuf::Message& proto)
{
    const string& dirname = GetDumpOptions()->graphDumpPath;
    Status status = env->RecursivelyCreateDir(dirname);
    if (!status.ok()) {
        LOG(WARNING) << "Failed to create " << dirname << " for dumping " << proto_type << ": " << status;
        return "(unavailable)";
    }
    string filepath = absl::StrCat(dirname, "/", name, ".pbtxt");
    if (env->FileExists(filepath).ok()) {
        status = env->DeleteFile(filepath);
    }
    status = WriteTextProto(env, filepath, proto);
    if (!status.ok()) {
        LOG(WARNING) << "Failed to dump " << proto_type << " to file: " << filepath << " : " << status;
        return "(unavailable)";
    }
    VLOG_LEVEL_2 << "Dumped " << proto_type << " to " << filepath;
    return filepath;
}

}  // anonymous namespace

string DumpGraphDefToFile(const string& name, GraphDef const& graph_def)
{
    return WriteTextProtoToUniqueFile(Env::Default(), name, "GraphDef", graph_def);
}

string DumpGraphToFile(const string& name, Graph const& graph, const FunctionLibraryDefinition* flib_def)
{
    GraphDef graph_def;
    graph.ToGraphDef(&graph_def);
    if (flib_def) {
        *graph_def.mutable_library() = flib_def->ToProto();
    }
    return DumpGraphDefToFile(name, graph_def);
}

string DumpFunctionDefToFile(const string& name, FunctionDef const& fdef)
{
    return WriteTextProtoToUniqueFile(Env::Default(), name, "FunctionDef", fdef);
}

}  // namespace dump_graph
}  // namespace npu_xla
}  // namespace tensorflow