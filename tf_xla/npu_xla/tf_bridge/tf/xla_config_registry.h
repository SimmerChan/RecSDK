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
/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef NPU_XLA_TF_BRIDGE_TF_XLA_CONFIG_REGISTRY_H_
#define NPU_XLA_TF_BRIDGE_TF_XLA_CONFIG_REGISTRY_H_

#include <functional>

#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {
namespace npu_xla {

namespace xla_config_registry {

// XlaGlobalJitLevel is used by XLA to expose its JIT level for processing
// single gpu and general (multi-gpu) graphs.
struct XlaGlobalJitLevel {
    OptimizerOptions::GlobalJitLevel single_gpu;
    OptimizerOptions::GlobalJitLevel general;
};

// Input is the jit_level in session config, and return value is the jit_level
// from XLA, reflecting the effect of the environment variable flags.
typedef std::function<XlaGlobalJitLevel(const OptimizerOptions::GlobalJitLevel&)> GlobalJitLevelGetterTy;

void RegisterGlobalJitLevelGetter(GlobalJitLevelGetterTy getter);

XlaGlobalJitLevel GetGlobalJitLevel(OptimizerOptions::GlobalJitLevel jit_level_in_session_opts);

#define REGISTER_XLA_CONFIG_GETTER(getter) REGISTER_XLA_CONFIG_GETTER_UNIQ_HELPER(__COUNTER__, getter)

#define REGISTER_XLA_CONFIG_GETTER_UNIQ_HELPER(ctr, getter) REGISTER_XLA_CONFIG_GETTER_UNIQ(ctr, getter)

#define REGISTER_XLA_CONFIG_GETTER_UNIQ(ctr, getter)     \
    static bool xla_config_registry_registration_##ctr = \
        (::tensorflow::npu_xla::xla_config_registry::RegisterGlobalJitLevelGetter(getter), true)

}  // namespace xla_config_registry

}  // namespace npu_xla
}  // namespace tensorflow

#endif  // NPU_XLA_TF_BRIDGE_TF_XLA_CONFIG_REGISTRY_H_
