/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: dataset ops.
 * Description: tf ops.
 * Author: S00548356
 * Create: 2024
 * History: NA
 */
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

    REGISTER_OP("RmaReadQueue")
    //.Input("input_tensor: float32")
    .Input("shm_addr: int64")
    .Output("output_tensor: float32")
    .Attr("output_types: type")
    .Attr("output_shapes: shape")
    .Attr("gshm: string")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    PartialTensorShape output_shapes;
    TF_RETURN_IF_ERROR(c->GetAttr("output_shapes", &output_shapes));

    shape_inference::ShapeHandle output_shape_handle;
    TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(
        output_shapes, &output_shape_handle));
    c->set_output(0, output_shape_handle);

    return tensorflow::Status::OK();
});

REGISTER_KERNEL_BUILDER(Name("RmaReadQueue").Device(DEVICE_CPU), CustOps);

REGISTER_OP("RmaWriteQueue")
.Input("input_tensor: float32")
.Input("shm_addr: int64")
.SetIsStateful()
.SetShapeFn(shape_inference::NoOutputs);

REGISTER_KERNEL_BUILDER(Name("RmaWriteQueue").Device(DEVICE_CPU), CustOps);

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
