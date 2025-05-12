#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace npu_hlo {

#define GEN_PASS_DEF_TFTOMHLOLEGALIZATIONPASS
#include "tf_mlir/mlir/npu_hlo/transforms/passes.h.inc"
namespace {

class TFReshapeToMHLO : public OpRewritePattern<mlir::TF::ReshapeOp> {
 public:
  using OpRewritePattern<mlir::TF::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::TF::ReshapeOp op,
                                PatternRewriter& rewriter) const override {
    // 获取输入和形状
    Value input = op.getTensor();
    Value shape = op.getShape();

    // 创建新的mhlo.DynamicReshape操作
    rewriter.replaceOpWithNewOp<mhlo::DynamicReshapeOp>(
        op, op.getResult().getType(), input, shape);

    return success();
  }
};

struct TFToMHLOLegalizationPass
    : public impl::TFToMHLOLegalizationPassBase<TFToMHLOLegalizationPass> {
  void runOnOperation() override {
    // 获取当前模块
    ModuleOp module = getOperation();
    MLIRContext* context = &getContext();

    // 设置重写模式
    RewritePatternSet patterns(context);
    patterns.add<TFReshapeToMHLO>(context);

    // 应用模式
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createTFToMHLOLegalizationPass() {
  return std::make_unique<TFToMHLOLegalizationPass>();
}

}  // namespace npu_hlo
}  // namespace mlir