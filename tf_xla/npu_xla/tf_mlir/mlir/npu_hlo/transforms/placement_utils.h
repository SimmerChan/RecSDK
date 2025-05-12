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

#ifndef NPU_XLA_TF_MLIR_MLIR_NPU_HLO_TRANSFORMS_PLACEMENT_UTILS_H_
#define NPU_XLA_TF_MLIR_MLIR_NPU_HLO_TRANSFORMS_PLACEMENT_UTILS_H_

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

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace placement_utils {
// Attrs for OP type
constexpr llvm::StringRef kShapeCalcAttr = "npu_hlo.shape_op";

// Attrs for placement
constexpr llvm::StringRef kPlaceAssignment = "npu_hlo.device";
constexpr llvm::StringRef kInputPlacementAttr = "input_placements";
constexpr llvm::StringRef kOutputPlacementAttr = "output_placements";
constexpr llvm::StringRef kCpu = "cpu";
constexpr llvm::StringRef kNpu = "npu";
constexpr llvm::StringRef kConst = "const";

enum class PlacementType { CPU, NPU, Const };
}  // namespace placement_utils
}  // namespace mlir

#endif  // NPU_XLA_TF_MLIR_MLIR_NPU_HLO_TRANSFORMS_PLACEMENT_UTILS_H_