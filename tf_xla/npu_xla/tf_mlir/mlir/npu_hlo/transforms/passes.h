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

#ifndef NPU_XLA_TF_MLIR_MLIR_NPU_HLO_TRANSFORMS_PASSES_H_
#define NPU_XLA_TF_MLIR_MLIR_NPU_HLO_TRANSFORMS_PASSES_H_

namespace mlir {

class ModuleOp;
template <typename T>
class OperationPass;

namespace npu_hlo {
// Replace const arguments to ConstOp and update argument type if it is a
// fixed-shaped input
std::unique_ptr<OperationPass<ModuleOp>> createReviseArgsForStaticRankPass();

// Legalize TensorFlow dialect to MHLO dialect
std::unique_ptr<OperationPass<ModuleOp>> createTFToMHLOLegalizationPass();
}  // namespace npu_hlo
}  // namespace mlir

#endif  // NPU_XLA_TF_MLIR_MLIR_NPU_HLO_TRANSFORMS_PASSES_H_