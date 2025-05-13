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

const char* const kXlaMustCompileAttr = "_NpuXlaMustCompile";

const char* const kXlaCompileAttr = "_NpuXlaCompile";

const char* const kReuseXlaCompileAttr = "_XlaCompile";

// User-provided through jit_scope APIs. Effective only when auto_jit is OFF.
const char* const kXlaScopeAttr = "_NpuXlaScope";

const char* const kReuseXlaScopeAttr = "_XlaScope";

// Automatically inserted by auto_jit to guide clustering results.  Effective
// only when auto_jit is ON.
const char* const kXlaInternalScopeAttr = "_NpuXlaInternalScope";

}  // namespace npu_xla
}  // namespace tensorflow