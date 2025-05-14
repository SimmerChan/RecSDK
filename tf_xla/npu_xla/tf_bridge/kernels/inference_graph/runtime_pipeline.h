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

#ifndef NPU_XLA_TF_BRIDGE_KERNELS_INFERENCE_GRAPH_RUNTIME_PIPELINE_H_
#define NPU_XLA_TF_BRIDGE_KERNELS_INFERENCE_GRAPH_RUNTIME_PIPELINE_H_

#include <memory>
#include <vector>

namespace DOPAI {
namespace GRT {
extern "C" {
std::vector<std::tuple<void*, uint64_t, const std::vector<int64_t>>> runtime_pipeline(
    const char* filePath, std::vector<void*> inputTensorPtr, std::vector<std::vector<long int>> shapeInfo,
    int outputNums);
}
}  // namespace GRT
}  // namespace DOPAI

#endif  // NPU_XLA_TF_BRIDGE_KERNELS_INFERENCE_GRAPH_RUNTIME_PIPELINE_H_
