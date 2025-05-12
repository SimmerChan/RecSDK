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

  // 主要接口：将GraphDef转换为StableHLO模块
  // 参数:
  //   graph_def: 输入的TensorFlow GraphDef
  //   config_proto: TensorFlow配置
  //   mlir_module: 输出的MLIR模块，包含StableHLO方言
  // 返回:
  //   转换是否成功的状态
  Status ConvertGraphdefToStablehlo(const CompilerInput& compiler_input,
                                    const std::string& output_path);

 private:
  // 将GraphDef转换为TF Executor Dialect
  Status ImportGraphDef(const CompilerInput& compiler_input);

  // 将TF Executor Dialect转换为TF Dialect
  Status ConvertTfExecutorToTf();

  // 将TF Dialect转换为StableHLO Dialect
  Status ConvertTfToStablehlo();

  // MLIR上下文
  std::unique_ptr<mlir::MLIRContext> context_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
};

}  // namespace npu_xla
}  // namespace tensorflow

#endif  // NPU_XLA_TF_BRIDGE_COMPILER_MLIR_CONVERTER_H_