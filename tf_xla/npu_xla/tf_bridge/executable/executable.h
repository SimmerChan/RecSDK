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

#ifndef NPU_XLA_TF_BRIDGE_EXECUTABLE_EXECUTABLE_H_
#define NPU_XLA_TF_BRIDGE_EXECUTABLE_EXECUTABLE_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/status.h"
#include "tf_bridge/compilation_result.pb.h"

namespace tensorflow {
namespace npu_xla {
using CompilationResult = CompilationResultProto;
class Executable {
public:
    explicit Executable(CompilationResult&& compilation_result);
    ~Executable();

    Status Run(OpKernelContext* ctx);

    Status DumpToFile(const std::string& filename) const;

private:
    std::unique_ptr<CompilationResult> compilation_result_;
    std::string compiled_model_path_;
    TF_DISALLOW_COPY_AND_ASSIGN(Executable);
};
}  // namespace npu_xla
}  // namespace tensorflow

#endif  // NPU_XLA_TF_BRIDGE_EXECUTABLE_EXECUTABLE_H_