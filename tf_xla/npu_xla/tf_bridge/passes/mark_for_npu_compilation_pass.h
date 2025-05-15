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
#ifndef NPU_XLA_TF_BRIDGE_PASSES_MARK_FOR_NPU_COMPILATION_PASS_H_
#define NPU_XLA_TF_BRIDGE_PASSES_MARK_FOR_NPU_COMPILATION_PASS_H_

#include "tensorflow/core/common_runtime/optimization_registry.h"

namespace tensorflow {
namespace npu_xla {
// The attribute that marks nodes to be grouped into functions by the
// encapsulate subgraphs pass.
extern const char* const kXlaClusterAttr;

// The attribute that marks nodes in a cluster to be placed outside the xla
// compilation by the encapsulate subgraphs pass.
extern const char* const kXlaOutsideCompilationAttr;

class MarkForNpuCompilationPass : public GraphOptimizationPass {
public:
    MarkForNpuCompilationPass() = default;
    Status Run(const GraphOptimizationPassOptions& options) override;
};

std::unordered_map<string, std::vector<string>>* GetWhitelistTable();
}  // namespace npu_xla
}  // namespace tensorflow

#endif  // NPU_XLA_TF_BRIDGE_PASSES_MARK_FOR_NPU_COMPILATION_PASS_H_
