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

#ifndef NPU_XLA_TF_BRIDGE_OPS_NPU_XLA_OPS_H_
#define NPU_XLA_TF_BRIDGE_OPS_NPU_XLA_OPS_H_

#include "tensorflow/core/framework/op.h"

namespace tensorflow {
REGISTER_OP("NpuXlaLaunch")
    .Input("constants: Tconstants")
    .Attr("Tconstants: list(type) >= 0")
    .Input("fixedshapes: Tfixedshapes")
    .Attr("Tfixedshapes: list(type) >= 0")
    .Input("hostargs: Thostargs")
    .Attr("Thostargs: list(type) >= 0")
    .Input("args: Targs")
    .Attr("Targs: list(type) >= 0")
    .Input("resources: Nresources * resource")
    .Attr("Nresources: int >= 0")
    .Output("hostresults: Thostresults")
    .Attr("Thostresults: list(type) >= 0")
    .Output("deviceresults: Tdeviceresults")
    .Attr("Tdeviceresults: list(type) >= 0")
    .Attr("mlir_function: func")
    // XLA random-number generation ops are stateful.
    .SetIsStateful()
    .Doc(R"(NpuXlaLaunchOp supports dynamic shape JIT. NpuXlaLaunchOp uses MLIR as the backend.)");
}

#endif  // NPU_XLA_TF_BRIDGE_OPS_NPU_XLA_OPS_H_