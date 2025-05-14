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

#ifndef NPU_XLA_TF_BRIDGE_KERNELS_INFERENCE_GRAPH_COMPILER_PIPELINE_H_
#define NPU_XLA_TF_BRIDGE_KERNELS_INFERENCE_GRAPH_COMPILER_PIPELINE_H_

#include <iostream>
#include <string>
#include <cstdlib>
#include <filesystem>

#include "absl/status/statusor.h"
#include "tsl/platform/logging.h"

namespace tensorflow {
namespace npu_xla {
namespace compiler {

const std::string BIN_MLIR_OPT = "inference-mlir-opt";

const std::string BIN_MLIR_COMPILER = "inference-mlir-compiler";

const std::string INFERENCE_GRAPH_PATH = "INFERENCE_GRAPH_PATH";

absl::Status RunSystemCommand(const std::string& command);

absl::Status CleanModelDirs(const std::string& path);

absl::Status CompiledInCodegenMode(const std::string& binPath, const std::string& modelPath);

absl::Status ExportModel(const std::string& binPath, const std::string& modelPath);

absl::Status RunPipeline(const std::string& modelPath);

}  // namespace compiler
}  // namespace npu_xla
}  // namespace tensorflow

#endif  // NPU_XLA_TF_BRIDGE_KERNELS_INFERENCE_GRAPH_COMPILER_PIPELINE_H_
