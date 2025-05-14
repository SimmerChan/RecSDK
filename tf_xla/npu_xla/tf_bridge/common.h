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

#ifndef NPU_XLA_TF_BRIDGE_COMMON_H_
#define NPU_XLA_TF_BRIDGE_COMMON_H_

#include <string>

#include "tensorflow/core/framework/tensor.h"
#include "tf_bridge/tf_compatible.h"

namespace tensorflow {

namespace npu_xla {

struct OptionalTensor {
    string name;           // A descriptive name
    bool present = false;  // Is the tensor present?
    Tensor value;          // If present, what is the Tensor's value?
};

struct TfBridgeOptions {
    // If tao bridge is enabled. Contrled by env var `BRIDGE_ENABLE_TAO` defaults
    // to false.
    bool enable_npu_xla;
    // Whether to enable functionalize control flow pass
    bool enable_control_flow;
    // Path to tf_mlir_bin.
    std::string tf_mlir_bin_path;
    // Path to cache.
    std::string cache_path;
    // Path to compilation product.
    std::string compilation_product_path;
};

// Get the globally singleton of TaoBridgeOptions.
// `force_refresh` is only used for debug. DO NOT update tao envs after
// it's initialized.
const TfBridgeOptions* GetTfBridgeOptions(bool force_refresh = false);

}  // namespace npu_xla
}  // namespace tensorflow

#endif  // NPU_XLA_TF_BRIDGE_COMMON_H_
