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
#ifndef NPU_XLA_TF_BRIDGE_COMPILER_MLIR_CONVERTER_H_
#define NPU_XLA_TF_BRIDGE_COMPILER_MLIR_CONVERTER_H_

#include "mlir/IR/BuiltinOps.h"
#include "tensorflow/core/platform/status.h"
#include "tf_mlir/compiler_input.pb.h"

namespace tensorflow {
namespace npu_xla {

// 该类实现从TensorFlow GraphDef到StableHLO的转换流程
// 转换路径: GraphDef -> TF Executor Dialect -> TF Dialect -> StableHLO Dialect
class MlirConverter {
public:
    MlirConverter();
    ~MlirConverter();

    Status ConvertGraphdefToStablehlo(const CompilerInput& compiler_input, const std::string& output_path);

private:
    // 将GraphDef转换为TF Executor Dialect
    Status ImportGraphDef(const CompilerInput& compiler_input);

    // 将TF Executor Dialect转换为TF Dialect
    Status ConvertTfExecutorToTf();

    // 将TF Dialect转换为StableHLO Dialect
    Status ConvertTfToStablehlo();

    // 设置PassManager的基本配置
    Status SetupPassManager(mlir::PassManager& pm, mlir::TimingScope& timing, bool prefer_tf2xla,
                            llvm::StringRef device_type);

    // 添加TensorFlow预处理相关的pass
    Status AddTFPreprocessingPasses(mlir::PassManager& pm);

    // 添加TensorFlow算子分解相关的pass
    Status AddTFDecompositionPasses(mlir::PassManager& pm);

    // 添加TensorFlow到MHLO转换相关的pass
    Status AddTFToMHLOPasses(mlir::PassManager& pm, llvm::StringRef device_type, bool prefer_tf2xla);

    // 添加MHLO到StableHLO转换相关的pass
    Status AddMHLOToStableHLOPasses(mlir::PassManager& pm);

    // MLIR上下文
    std::unique_ptr<mlir::MLIRContext> context_;
    mlir::OwningOpRef<mlir::ModuleOp> module_;
};

}  // namespace npu_xla
}  // namespace tensorflow

#endif  // NPU_XLA_TF_BRIDGE_COMPILER_MLIR_CONVERTER_H_