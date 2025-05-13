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

#ifndef NPU_XLA_TF_BRIDGE_UTILS_DUMP_OPTIONS_H_
#define NPU_XLA_TF_BRIDGE_UTILS_DUMP_OPTIONS_H_

#include <string>

namespace tensorflow {
namespace npu_xla {
struct DumpOptions {
    // Path prefix to dump output graph of passes. Contrled by env var
    // `NPU_XLA_GRAPH_DUMP_PATH` defaults to '/tmp/npu_xla'.
    std::string graphDumpPath;
};

// Get the globally singleton of DumpOptions.
const DumpOptions* GetDumpOptions();
}  // namespace npu_xla
}  // namespace tensorflow

#endif  // NPU_XLA_TF_BRIDGE_UTILS_DUMP_OPTIONS_H_
