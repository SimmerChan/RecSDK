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

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Function.h"  // TF:llvm-project
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Pass/Pass.h"              // TF:llvm-project
#include "mlir/Support/LogicalResult.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tf_mlir/mlir/npu_hlo/npu_hlo_utils.h"
#include "tf_mlir/mlir/npu_hlo/transforms/placement_utils.h"

namespace mlir {

using npu_hlo::kHloInputShapeAttr;
using npu_hlo::kHloInputValueAttr;
using placement_utils::kConst;
using placement_utils::kInputPlacementAttr;

namespace npu_hlo {
#define GEN_PASS_DEF_REVISEARGUMENTSFORSTATICRANKPASS
#include "tf_mlir/mlir/npu_hlo/transforms/passes.h.inc"
namespace {
constexpr int SMALL_VECTOR_DEFAULT_SIZE = 4;

// Replace const arguments to ConstOp and update argument type if it is a
// fixed-shaped input
struct ReviseArgsForStaticRankPass : public impl::ReviseArgumentsForStaticRankPassBase<ReviseArgsForStaticRankPass> {
    explicit ReviseArgsForStaticRankPass()
        : impl::ReviseArgumentsForStaticRankPassBase<
              ReviseArgsForStaticRankPass>::ReviseArgumentsForStaticRankPassBase()
    {
    }

    void runOnOperation() override;
    void replaceArgWithConstOp(func::FuncOp main, DictionaryAttr dict_attr, unsigned idx);
    void updateInputTypesAndAttributes(func::FuncOp main_func, DictionaryAttr dict_attr,
                                       SmallVector<StringRef, SMALL_VECTOR_DEFAULT_SIZE>& new_input_placements);
};

void ReviseArgsForStaticRankPass::replaceArgWithConstOp(func::FuncOp main, DictionaryAttr dict_attr, unsigned idx)
{
    ModuleOp module = getOperation();
    auto attr = dict_attr.get((kHloInputValueAttr + ("_" + llvm::Twine(idx))).str());
    if (!attr) {
        module.emitError() << "kHloInputValueAttr not found for idx: " << idx << ".\n";
        return signalPassFailure();
    }
    OpBuilder b(&main.front().front());
    Value const_input = b.create<TF::ConstOp>(main.getLoc(), attr);
    main.getArgument(idx).replaceAllUsesWith(const_input);
    // Here we will not erase the unused argument, since the arg indices in ctx
    // has to match the arg indices of the Mlir Module when ral recv inputs
}

// 新增函数：处理输入类型和属性更新
void ReviseArgsForStaticRankPass::updateInputTypesAndAttributes(
    func::FuncOp main_func, DictionaryAttr dict_attr,
    SmallVector<StringRef, SMALL_VECTOR_DEFAULT_SIZE>& new_input_placements)
{
    ModuleOp module = getOperation();
    auto num_inputs = main_func.getNumArguments();

    // 更新输入类型
    auto func_type = main_func.getFunctionType();
    SmallVector<Type, SMALL_VECTOR_DEFAULT_SIZE> input_types(func_type.getInputs().begin(),
                                                             func_type.getInputs().end());
    if (input_types.size() != num_inputs) {
        module.emitError("Error: input_types.size() is not equal to num of inputs.\n");
        return signalPassFailure();
    }
    for (int i = 0; i < num_inputs; ++i) {
        auto attr = dict_attr.get((kHloInputShapeAttr + ("_" + llvm::Twine(i))).str());
        if (attr) {
            auto type = attr.cast<DenseElementsAttr>().getType();
            main_func.getArgument(i).setType(type);
            input_types[i] = type;
        }
    }
    OpBuilder builder(&main_func.front().front());
    auto new_func_type = builder.getFunctionType(input_types, func_type.getResults());
    main_func.setType(new_func_type);

    // 更新输入位置属性
    SmallVector<mlir::NamedAttribute, SMALL_VECTOR_DEFAULT_SIZE> new_attributes;
    for (auto attr : dict_attr) {
        if (attr.getName() != placement_utils::kInputPlacementAttr) {
            new_attributes.push_back(attr);
        }
    }
    new_attributes.push_back(
        builder.getNamedAttr("input_placements", builder.getStringAttr(llvm::join(new_input_placements, ","))));
    main_func->setAttr("tf.entry_function", builder.getDictionaryAttr(new_attributes));
}

void ReviseArgsForStaticRankPass::runOnOperation()
{
    ModuleOp module = getOperation();
    func::FuncOp main_func = module.lookupSymbol<func::FuncOp>("main");
    if (!main_func) {
        module.emitError("Error: main_func not found.\n");
        return signalPassFailure();
    }
    auto num_inputs = main_func.getNumArguments();
    auto dict_attr = main_func->getAttrOfType<DictionaryAttr>("tf.entry_function");
    if (!dict_attr) {
        return;
    }
    auto input_placements_attr = dict_attr.get(placement_utils::kInputPlacementAttr);
    if (!input_placements_attr) {
        return;
    }
    SmallVector<StringRef, SMALL_VECTOR_DEFAULT_SIZE> input_placements;
    input_placements_attr.cast<mlir::StringAttr>().getValue().split(input_placements, ',', -1, false);
    if (input_placements.size() != num_inputs) {
        module.emitError("Error: input_placements.size() is not equal to num of inputs.\n");
        return signalPassFailure();
    }
    SmallVector<StringRef, SMALL_VECTOR_DEFAULT_SIZE> new_input_placements;

    // 为每个常量输入创建ConstOp并替换参数
    for (int i = 0; i < num_inputs; ++i) {
        auto placement = input_placements[i];
        if (placement == kConst) {
            // It makes no difference whether cpu or gpu for const args here,
            // since the arg will have no consumer. However, let's just leave
            // it as gpu
            new_input_placements.push_back(placement_utils::kNpu);
            replaceArgWithConstOp(main_func, dict_attr, i);
        } else {
            new_input_placements.push_back(placement);
        }
    }

    // 更新输入类型和属性
    updateInputTypesAndAttributes(main_func, dict_attr, new_input_placements);
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createReviseArgsForStaticRankPass()
{
    return std::make_unique<ReviseArgsForStaticRankPass>();
}

}  // namespace npu_hlo
}  // namespace mlir
