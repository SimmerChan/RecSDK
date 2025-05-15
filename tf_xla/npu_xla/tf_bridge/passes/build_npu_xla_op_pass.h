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
// Copyright 2021 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NPU_XLA_TF_BRIDGE_PASSES_BUILD_NPU_XLA_OP_H_
#define NPU_XLA_TF_BRIDGE_PASSES_BUILD_NPU_XLA_OP_H_

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/lib/core/status.h"
#include "tf_bridge/common.h"

namespace tensorflow {
namespace npu_xla {

class BuildNpuXlaOpPass : public GraphOptimizationPass {
public:
    Status Run(const GraphOptimizationPassOptions& options) override;

private:
    bool inner_launch_{true};  // inner attribute of NpuXlaLaunchOp op.
};

}  // namespace npu_xla
}  // namespace tensorflow

#endif  // NPU_XLA_TF_BRIDGE_PASSES_BUILD_NPU_XLA_OP_H_