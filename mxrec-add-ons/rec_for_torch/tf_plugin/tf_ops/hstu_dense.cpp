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

using namespace tensorflow;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

using namespace std;
using namespace chrono;

using OpKernelConstructionPtr = OpKernelConstruction*;
using OpKernelContextPtr = OpKernelContext*;
using InferenceContextPtr = ::tensorflow::shape_inference::InferenceContext*;

namespace {
class HstuDenseForward : public OpKernel {
public:
    explicit HstuDenseForward(OpKernelConstructionPtr context) : OpKernel(context) {}

    void Compute(OpKernelContextPtr context) override
    {
        std::cout << "HstuDenseForward not installed!!" << std::endl;
    }

    ~HstuDenseForward() override = default;
};

class HstuDenseBackward : public OpKernel {
public:
    explicit HstuDenseBackward(OpKernelConstructionPtr context) : OpKernel(context) {}

    void Compute(OpKernelContextPtr context) override
    {
        std::cout << "HstuDenseBackward not installed!!" << std::endl;
    }

    ~HstuDenseBackward() override = default;
};
}  // namespace


namespace tensorflow {
REGISTER_OP("HstuDenseForward")
    .Input("q: T")
    .Input("k: T")
    .Input("v: T")
    .Input("mask: T")
    .Input("attn_bias: T")
    .Output("atten_output: T")
    .Attr("maskType: int")
    .Attr("max_seq_len: int")
    .Attr("silu_scale: float")
    .Attr("layout: string")
    .Attr("seq_offsets: list(int)")
    .Attr("T: {float16, float32, bfloat16}")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ShapeHandle q_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &q_shape));

        tensorflow::shape_inference::DimensionHandle batch_size = c->Dim(q_shape, 0);
        tensorflow::shape_inference::DimensionHandle seq = c->Dim(q_shape, 1);
        tensorflow::shape_inference::DimensionHandle num_heads = c->Dim(q_shape, 2);
        tensorflow::shape_inference::DimensionHandle attn_emb = c->Dim(q_shape, 3);

        int64_t shape0 = c->Value(batch_size);
        int64_t shape1 = c->Value(seq);
        int64_t shape2 = c->Value(num_heads);
        int64_t shape3 = c->Value(attn_emb);

        c->set_output(0, c->MakeShape({shape0, shape1, shape2, shape3}));
        return Status::OK();
    });
REGISTER_KERNEL_BUILDER(Name("HstuDenseForward").Device(DEVICE_CPU), HstuDenseForward)

REGISTER_OP("HstuDenseBackward")
    .Input("grad: T")
    .Input("q: T")
    .Input("k: T")
    .Input("v: T")
    .Input("mask: T")
    .Input("attn_bias: T")
    .Output("q_grad: T")
    .Output("k_grad: T")
    .Output("v_grad: T")
    .Output("attn_bias_grad: T")
    .Attr("layout: string")
    .Attr("mask_type: int")
    .Attr("max_seq_len: int")
    .Attr("silu_scale: float")
    .Attr("seq_offsets: list(int)")
    .Attr("T: {float16, float32, bfloat16}")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ShapeHandle q_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &q_shape));

        tensorflow::shape_inference::DimensionHandle batch_size = c->Dim(q_shape, 0);
        tensorflow::shape_inference::DimensionHandle seq = c->Dim(q_shape, 1);
        tensorflow::shape_inference::DimensionHandle num_heads = c->Dim(q_shape, 2);
        tensorflow::shape_inference::DimensionHandle attn_emb = c->Dim(q_shape, 3);

        int64_t shape0 = c->Value(batch_size);
        int64_t shape1 = c->Value(seq);
        int64_t shape2 = c->Value(num_heads);
        int64_t shape3 = c->Value(attn_emb);

        c->set_output(0, c->MakeShape({shape0, shape1, shape2, shape3}));
        c->set_output(1, c->MakeShape({shape0, shape1, shape2, shape3}));
        c->set_output(2, c->MakeShape({shape0, shape1, shape2, shape3}));
        c->set_output(3, c->MakeShape({shape0, shape2, shape1, shape1}));
        return Status::OK();
    });
REGISTER_KERNEL_BUILDER(Name("HstuDenseBackward").Device(DEVICE_CPU), HstuDenseBackward)

}  // namespace tensorflow