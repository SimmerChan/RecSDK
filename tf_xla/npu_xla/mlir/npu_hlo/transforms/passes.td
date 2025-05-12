/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

include "mlir/Pass/PassBase.td"

def ReviseArgumentsForStaticRankPass : Pass<"npu-tf-revise-args-for-static-rank", "ModuleOp"> {
  let summary = "extract the const/fixed-shape input info from jit static rank compilation cache, and update the info in the IR.";
  let constructor = "createReviseArgsForStaticRankPass()";
}

def TFToMHLOLegalizationPass : Pass<"npu-tf-to-mhlo", "ModuleOp"> {
  let summary = "Legalize TensorFlow dialect to MHLO dialect.";
  let description = [{
    This pass legalizes TensorFlow operations to MHLO operations.
    It handles both standard TensorFlow ops and custom ops.
  }];
  let constructor = "createTFToMHLOLegalizationPass()";
}