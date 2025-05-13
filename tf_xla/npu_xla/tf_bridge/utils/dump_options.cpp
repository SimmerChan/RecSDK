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

#include "tf_bridge/utils/dump_options.h"

#include <mutex>

#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace npu_xla {
namespace {
static DumpOptions* dumpOptions = nullptr;
static std::once_flag initFlag;
static std::mutex mut;

static void InitDumpOptions()
{
    std::lock_guard<std::mutex> lock(mut);
    if (!dumpOptions) {
        dumpOptions = new DumpOptions();
    }

    {
        TF_CHECK_OK(ReadStringFromEnvVar("NPU_XLA_GRAPH_DUMP_PATH", "/tmp/npu_xla", &dumpOptions->graphDumpPath));
    }
}
}  // namespace

const DumpOptions* GetDumpOptions()
{
    std::call_once(initFlag, &InitDumpOptions);
    return dumpOptions;
}
}  // namespace npu_xla
}  // namespace tensorflow