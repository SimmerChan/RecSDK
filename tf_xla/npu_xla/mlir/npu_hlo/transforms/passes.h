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