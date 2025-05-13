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

#include "tf_bridge/common.h"

#include <climits>
#include <mutex>

#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace npu_xla {

namespace {

static TfBridgeOptions* opts{nullptr};
static std::once_flag opt_init;
static std::mutex opt_mtx_;
static void AllocateTfBridgeFlags() {
  std::lock_guard<std::mutex> lock(opt_mtx_);
  if (!opts) {
    opts = new TfBridgeOptions();
  }
  TF_CHECK_OK(
      ReadBoolFromEnvVar("TF_ENABLE_NPU_XLA", true, &opts->enable_npu_xla));
  if (!opts->enable_npu_xla) {
    LOG(WARNING) << "NPU XLA compiler disabled, set env var TF_ENABLE_NPU_XLA "
                 << "to true to enable NPU XLA compiler.";
  }
  TF_CHECK_OK(ReadBoolFromEnvVar("NPU_XLA_ENABLE_CONTROL_FLOW", false,
                                 &opts->enable_control_flow));
  TF_CHECK_OK(ReadStringFromEnvVar("TF_MLIR_BIN_PATH", "tf_mlir_main",
                                   &opts->tf_mlir_bin_path));
  TF_CHECK_OK(ReadStringFromEnvVar("CACHE_PATH", "", &opts->cache_path));
  TF_CHECK_OK(ReadStringFromEnvVar("COMPILE_PRODUCT_PATH", "/tmp",
                                   &opts->compilation_product_path));
}

}  // namespace

const TfBridgeOptions* GetTfBridgeOptions(bool force_refresh) {
  std::call_once(opt_init, &AllocateTfBridgeFlags);
  if (force_refresh) {
    AllocateTfBridgeFlags();
  }
  return opts;
}

}  // namespace npu_xla
}  // namespace tensorflow
