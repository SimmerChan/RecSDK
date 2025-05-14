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

#include <string>

#include "tensorflow/core/framework/op_kernel.h"
#include "tf_bridge/tf/log.h"

namespace tensorflow {
const char* const DEVICE_NPU = "NPU";
using ::tensorflow::npu_xla::VLOG_LEVEL_1;

template <std::string dummyOp>
class NpuDummyOp : public OpKernel {
public:
    explicit NpuDummyOp(OpKernelConstruction* ctx) : OpKernel(ctx), name_(dummyOp)
    {
        VLOG(VLOG_LEVEL_1) << "Construct NPU dummy op: " << name_;
    }

    void Compute(OpKernelContext* ctx) override
    {
        OP_REQUIRES_OK(ctx, errors::Internal("Computing NPU dummy op: ", name_));
    }

private:
    std::string name_;
};

#define REGISTER_DUMMY_OP(op) REGISTER_KERNEL_BUILDER(Name(#op).Device(DEVICE_NPU), NpuDummyOp<#op>);

REGISTER_DUMMY_OP(AddV2);
REGISTER_DUMMY_OP(Cast);
REGISTER_DUMMY_OP(Maximum);
REGISTER_DUMMY_OP(Minimum);
REGISTER_DUMMY_OP(RandomStandardNormal);
REGISTER_DUMMY_OP(Mul);
REGISTER_DUMMY_OP(RandomUniformInt);
REGISTER_DUMMY_OP(RandomUniform);
REGISTER_DUMMY_OP(Sub);
REGISTER_DUMMY_OP(Range);
REGISTER_DUMMY_OP(Max);
REGISTER_DUMMY_OP(ExpandDims);
REGISTER_DUMMY_OP(Less);
REGISTER_DUMMY_OP(LessEqual);
REGISTER_DUMMY_OP(StridedSlice);
REGISTER_DUMMY_OP(Pack);
REGISTER_DUMMY_OP(Unpack);
REGISTER_DUMMY_OP(Reshape);
REGISTER_DUMMY_OP(RealDiv);
REGISTER_DUMMY_OP(Sqrt);
REGISTER_DUMMY_OP(TruncatedNormal);
REGISTER_DUMMY_OP(MatMul);
REGISTER_DUMMY_OP(Transpose);
REGISTER_DUMMY_OP(BatchMatMulV2);
REGISTER_DUMMY_OP(Sigmoid);
REGISTER_DUMMY_OP(IdentityN);
REGISTER_DUMMY_OP(AssignVariableOp);
REGISTER_DUMMY_OP(ReadVariableOp);
REGISTER_DUMMY_OP(Mean);
REGISTER_DUMMY_OP(StopGradient);
REGISTER_DUMMY_OP(SquaredDifference);
REGISTER_DUMMY_OP(Rsqrt);
REGISTER_DUMMY_OP(BiasAdd);
REGISTER_DUMMY_OP(ConcatV2);
REGISTER_DUMMY_OP(FloorDiv);
REGISTER_DUMMY_OP(Tile);
REGISTER_DUMMY_OP(Neg);
REGISTER_DUMMY_OP(Relu);
REGISTER_DUMMY_OP(Abs);
REGISTER_DUMMY_OP(Squeeze);
REGISTER_DUMMY_OP(Greater);
REGISTER_DUMMY_OP(GreaterEqual);
REGISTER_DUMMY_OP(Select);
REGISTER_DUMMY_OP(Exp);
REGISTER_DUMMY_OP(Log1p);
REGISTER_DUMMY_OP(Equal);
REGISTER_DUMMY_OP(Merge);
REGISTER_DUMMY_OP(Fill);
REGISTER_DUMMY_OP(BroadcastGradientArgs);
REGISTER_DUMMY_OP(Sum);
REGISTER_DUMMY_OP(Shape);
REGISTER_DUMMY_OP(Reciprocal);
REGISTER_DUMMY_OP(AddN);
REGISTER_DUMMY_OP(BiasAddGrad);
REGISTER_DUMMY_OP(ReluGrad);
REGISTER_DUMMY_OP(Sign);
REGISTER_DUMMY_OP(RsqrtGrad);
REGISTER_DUMMY_OP(FloorMod);
REGISTER_DUMMY_OP(ConcatOffset);
REGISTER_DUMMY_OP(Slice);
REGISTER_DUMMY_OP(StridedSliceGrad);
REGISTER_DUMMY_OP(ZerosLike);
REGISTER_DUMMY_OP(InvertPermutation);
REGISTER_DUMMY_OP(ApplyAdam);
REGISTER_DUMMY_OP(ResourceApplyAdam);
REGISTER_DUMMY_OP(LogicalOr);
REGISTER_DUMMY_OP(LogicalNot);
REGISTER_DUMMY_OP(Prod);
REGISTER_DUMMY_OP(Where);
REGISTER_DUMMY_OP(GatherV2);
REGISTER_DUMMY_OP(Split);
REGISTER_DUMMY_OP(SplitV);
REGISTER_DUMMY_OP(OneHot);
REGISTER_DUMMY_OP(Pow);
REGISTER_DUMMY_OP(Tanh);
REGISTER_DUMMY_OP(Pad);
}  // namespace tensorflow