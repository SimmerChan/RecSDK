/* Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.

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

#include <algorithm>
#include <atomic>
#include <map>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"

using namespace tensorflow;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

using namespace std;
using namespace chrono;

using OpKernelConstructionPtr = OpKernelConstruction*;
using OpKernelContextPtr = OpKernelContext*;
using InferenceContextPtr = ::tensorflow::shape_inference::InferenceContext*;

namespace {
    class CustOps : public OpKernel {
    public:
        explicit CustOps(OpKernelConstructionPtr context) : OpKernel(context)
        {
        }

        void Compute(OpKernelContextPtr context) override
        {
            std::cout << " Cust opp not installed!!" << std::endl;
        }

        ~CustOps() override = default;
    };
}

namespace tensorflow {

REGISTER_OP("RmaSwap")
.Input("update_table: float32")
.Input("update_index: int64")
.Output("output: int64")
.Attr("shm_swap_in: string")
.Attr("shm_swap_out: string")
.SetIsStateful()
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
c->set_output(0, c->MakeShape({8}));
return tensorflow::Status::OK();
});

REGISTER_KERNEL_BUILDER(Name("RmaSwap").Device(DEVICE_CPU), CustOps);

REGISTER_OP("RmaSwapMultiTables")
.Input("table0: float32")
.Input("table1: float32")
.Input("table2: float32")
.Input("swap_in_index: int64")
.Input("swap_out_index: int64")
.Output("output: int64")
.Attr("shm_swap_in: string")
.Attr("shm_swap_out: string")
.SetIsStateful()
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
c->set_output(0, c->MakeShape({48}));
return tensorflow::Status::OK();
});

REGISTER_KERNEL_BUILDER(Name("RmaSwapMultiTables").Device(DEVICE_CPU), CustOps);
}
