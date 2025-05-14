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

#include "compiler_pipeline.h"

namespace tensorflow {
namespace npu_xla {
namespace compiler {

absl::Status RunSystemCommand(const std::string& command)
{
    VLOG(1) << "Runing command: " << command;
    auto ret = std::system(command.c_str());
    if (ret != 0) {
        VLOG(1) << "Command failed: " << command << ", errno: " << ret;
        return absl::InternalError("Compiler command failed");
    }
    return absl::OkStatus();
}

absl::Status CleanModelDirs(const std::string& path)
{
    VLOG(1) << "CleanModelDirs path: " << path;
    if (!std::filesystem::exists(path)) {
        return absl::InvalidArgumentError(path);
    }

    std::string kernelsPath = path + "/kernels";
    std::filesystem::remove_all(kernelsPath);
    std::filesystem::create_directory(kernelsPath);
    std::filesystem::remove_all(path + "/model_before_gendag.mlir");
    std::filesystem::remove_all(path + "/model_hivm.mlir");
    std::filesystem::remove_all(path + "/model.fb");

    return absl::OkStatus();
}

absl::Status CompiledInCodegenMode(const std::string& binPath, const std::string& modelPath)
{
    std::string cmd;

    cmd = binPath + "/" + BIN_MLIR_OPT +
          " --stablehlo-to-hfusion-pipeline"
          " --lower-hfusion-pipeline="
          "\"multi-kernel=true block-dim=40 fusion-output-mode=multi max-horizontal-fusion-size=0\""
          " --convert-hfusion-to-hivm " +
          modelPath + "/model.stablehlo" + " -o " + modelPath + "/model_before_gendag.mlir";
    auto ret = RunSystemCommand(cmd);
    if (ret != absl::OkStatus()) {
        return ret;
    }

    cmd = binPath + "/" + BIN_MLIR_OPT + " --generate-runtime-dag=graphName=model " + modelPath +
          "/model_before_gendag.mlir -o " + modelPath + "/model_hivm.mlir";
    ret = RunSystemCommand(cmd);
    if (ret != absl::OkStatus()) {
        return ret;
    }

    cmd = binPath + "/" + BIN_MLIR_COMPILER +
          " --enable-hfusion-compile=false"
          " --enable-hivm-compile=true"
          " --enable-lir-compile=true " +
          modelPath + "/model_hivm.mlir -o " + modelPath + "/model";
    return RunSystemCommand(cmd);
}

absl::Status ExportModel(const std::string& binPath, const std::string& modelPath)
{
    auto ret = CompiledInCodegenMode(binPath, modelPath);
    if (ret != absl::OkStatus()) {
        return ret;
    }

    std::string oPath = modelPath + "/model.o";
    std::string soPath = modelPath + "/libmodel.so";
    std::string kernelsPath = modelPath + "/kernels";

    std::string copyCmd = "mv " + oPath + " " + soPath + " " + kernelsPath;
    ret = RunSystemCommand(copyCmd);
    if (ret != absl::OkStatus()) {
        return ret;
    }

    std::string moveCmd = "mv model.fb " + modelPath + "/";
    return RunSystemCommand(moveCmd);
}

absl::Status RunPipeline(const std::string& modelPath)
{
    char* binPath = std::getenv(INFERENCE_GRAPH_PATH.c_str());
    if (binPath == nullptr) {
        std::string errMsg = "INFERENCE_GRAPH_PATH is not set, user need to export path to inference graph binary,"
                             " which include " +
                             BIN_MLIR_OPT + "/" + BIN_MLIR_COMPILER;
        return absl::InvalidArgumentError(errMsg);
    }

    auto ret = ExportModel(binPath, modelPath);
    if (ret != absl::OkStatus()) {
        return ret;
    }

    return absl::OkStatus();
}

}  // namespace compiler
}  // namespace npu_xla
}  // namespace tensorflow
