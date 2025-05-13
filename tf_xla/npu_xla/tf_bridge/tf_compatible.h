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

#ifndef NPU_XLA_TF_BRIDGE_TF_COMPATIBLE_H_
#define NPU_XLA_TF_BRIDGE_TF_COMPATIBLE_H_

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/public/version.h"

// defs moved or renamed between headers in different tf versions
// for now, just for tf2
// #if TF_MAJOR_VERSION > 1
// TF2.4
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_def_builder_util.h"
#include "tensorflow/core/graph/graph_node_util.h"
// #else
// TF1.12, TF1.15
// #include "tensorflow/core/graph/graph_constructor.h"
// #include "tensorflow/core/graph/graph_def_builder_util.h"
// #endif

// #if TF_MAJOR_VERSION > 1 || TF_MINOR_VERSION > 12
// TF1.15, TF2.4
#include "tensorflow/core/framework/bounds_check.h"
// #else
// TF1.12
// #include "tensorflow/core/kernels/bounds_check.h"
// #endif

#endif  // NPU_XLA_TF_BRIDGE_TF_COMPATIBLE_H_
