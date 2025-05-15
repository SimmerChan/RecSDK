/**
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
/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

// Adopted from tensorflow/core/common_runtime/lower_while_op.h

#ifndef NPU_XLA_TF_BRIDGE_TF_LOWER_WHILE_OP_H_
#define NPU_XLA_TF_BRIDGE_TF_LOWER_WHILE_OP_H_

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace npu_xla {

// Replaces While node `n` with its lowered form that uses Enter, Exit, Switch,
// Merge, NextIteration and LoopCond nodes.
Status RewriteWhileNode(Node* n, Graph* g, const FunctionLibraryDefinition& flib);

}  // namespace npu_xla
}  // namespace tensorflow

#endif  // NPU_XLA_TF_BRIDGE_TF_LOWER_WHILE_OP_H_