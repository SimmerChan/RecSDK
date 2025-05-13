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
//===- VectorToGPU.cpp - Convert vector to GPU dialect ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of vector operations to GPU dialect ops.
//
//===----------------------------------------------------------------------===//

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

struct TFReshapeToMHLO : public OpRewritePattern<mlir::TF::ReshapeOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(mlir::TF::ReshapeOp op, PatternRewriter& rewriter) const override
    {
        // 获取输入和形状
        Value input = op.getTensor();
        Value shape = op.getShape();

        // 创建新的mhlo.DynamicReshape操作
        rewriter.replaceOpWithNewOp<mhlo::DynamicReshapeOp>(op, op.getResult().getType(), input, shape);

        return success();
    }
};

struct TFToMHLOLegalizationPass : public impl::TFToMHLOLegalizationPassBase<TFToMHLOLegalizationPass> {
    void runOnOperation() override
    {
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

std::unique_ptr<OperationPass<ModuleOp>> createTFToMHLOLegalizationPass()
{
    return std::make_unique<TFToMHLOLegalizationPass>();
}

}  // namespace npu_hlo
}  // namespace mlir