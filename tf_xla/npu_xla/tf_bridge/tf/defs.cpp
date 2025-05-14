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

#include "tf_bridge/tf/defs.h"

namespace tensorflow {
namespace npu_xla {

const char* const K_XLA_MUST_COMPILE_ATTR = "_NpuXlaMustCompile";

const char* const K_XLA_COMPILE_ATTR = "_NpuXlaCompile";

const char* const K_REUSE_XLA_COMPILE_ATTR = "_XlaCompile";

// User-provided through jit_scope APIs. Effective only when auto_jit is OFF.
const char* const K_XLA_SCOPE_ATTR = "_NpuXlaScope";

const char* const K_REUSE_XLA_SCOPE_ATTR = "_XlaScope";

// Automatically inserted by auto_jit to guide clustering results.  Effective
// only when auto_jit is ON.
const char* const K_XLA_INTERNAL_SCOPE_ATTR = "_NpuXlaInternalScope";

}  // namespace npu_xla
}  // namespace tensorflow