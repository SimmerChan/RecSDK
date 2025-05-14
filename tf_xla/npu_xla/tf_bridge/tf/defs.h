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

#ifndef NPU_XLA_TF_BRIDGE_TF_DEFS_H_
#define NPU_XLA_TF_BRIDGE_TF_DEFS_H_

namespace tensorflow {
namespace npu_xla {

// Name of attribute used to tag operators for compilation with XLA

// Implies must-compile semantics: either it will be compiled
// with XLA, or an error will be thrown.
extern const char* const K_XLA_MUST_COMPILE_ATTR;

// Implies auto-clustering: tagged nodes will be clustered and compiled with XLA
// on a best-effort basis.
extern const char* const K_XLA_COMPILE_ATTR;
extern const char* const K_REUSE_XLA_COMPILE_ATTR;

// Implies auto-clustering within the given scope.
extern const char* const K_XLA_SCOPE_ATTR;
extern const char* const K_REUSE_XLA_SCOPE_ATTR;

extern const char* const K_XLA_INTERNAL_SCOPE_ATTR;

}  // namespace npu_xla
}  // namespace tensorflow

#endif  // NPU_XLA_TF_BRIDGE_TF_DEFS_H_
