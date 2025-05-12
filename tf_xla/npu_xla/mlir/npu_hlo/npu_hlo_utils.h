#ifndef NPU_XLA_TF_MLIR_MLIR_NPU_HLO_NPU_HLO_UTILS_H_
#define NPU_XLA_TF_MLIR_MLIR_NPU_HLO_NPU_HLO_UTILS_H_

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace npu_hlo {

constexpr llvm::StringRef kHloInputShapeAttr = "npu_hlo.input_shape";
constexpr llvm::StringRef kHloInputValueAttr = "npu_hlo.input_value";

}  // namespace npu_hlo
}  // namespace mlir

#endif  // NPU_XLA_TF_MLIR_MLIR_NPU_HLO_NPU_HLO_UTILS_H_