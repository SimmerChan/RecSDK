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

#include "common_hdrs/types.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
const char* const DEVICE_NPU = "NPU";
using ::tensorflow::npu_xla::VLOG_LEVEL_1;

enum class DummyOpT {
    Const = 0,
    RandomStandardNormal,
    Mul,
    AddV2,
    RandomUniformInt,
    RandomUniform,
    Max,
    Maximum,
    Minimum,
    Range,
    ExpandDims,
    Cast,
    Less,
    LessEqual,
    StridedSlice,
    Pack,
    Unpack,
    Reshape,
    RealDiv,
    Sqrt,
    TruncatedNormal,
    VariableV2,
    Assign,
    Identity,
    MatMul,
    Transpose,
    BatchMatMulV2,
    Sigmoid,
    IdentityN,
    VarHandleOp,
    VarIsInitializedOp,
    AssignVariableOp,
    ReadVariableOp,
    Mean,
    StopGradient,
    SquaredDifference,
    Rsqrt,
    Sub,
    BiasAdd,
    ConcatV2,
    FloorDiv,
    Tile,
    Neg,
    Relu,
    Abs,
    Squeeze,
    Greater,
    GreaterEqual,
    Select,
    Exp,
    Log1p,
    Equal,
    Switch,
    Merge,
    Fill,
    NoOp,
    BroadcastGradientArgs,
    Sum,
    Shape,
    Reciprocal,
    AddN,
    BiasAddGrad,
    ReluGrad,
    Sign,
    RsqrtGrad,
    FloorMod,
    ConcatOffset,
    Slice,
    StridedSliceGrad,
    ZerosLike,
    InvertPermutation,
    ApplyAdam,
    ResourceApplyAdam,
    Snapshot,
    LogicalOr,
    LogicalNot,
    Prod,
    Where,
    GatherV2,
    Split,
    SplitV,
    OneHot,
    Pow,
    Softmax,
    Tanh,
    Pad,
};

std::string DummyOpTToString(DummyOpT op)
{
    switch (op) {
        case DummyOpT::Const:
            return "Const";
        case DummyOpT::RandomStandardNormal:
            return "RandomStandardNormal";
        case DummyOpT::Mul:
            return "Mul";
        case DummyOpT::AddV2:
            return "AddV2";
        case DummyOpT::RandomUniformInt:
            return "RandomUniformInt";
        case DummyOpT::RandomUniform:
            return "RandomUniform";
        case DummyOpT::Max:
            return "Max";
        case DummyOpT::Maximum:
            return "Maximum";
        case DummyOpT::Minimum:
            return "Minimum";
        case DummyOpT::Range:
            return "Range";
        case DummyOpT::ExpandDims:
            return "ExpandDims";
        case DummyOpT::Cast:
            return "Cast";
        case DummyOpT::Less:
            return "Less";
        case DummyOpT::LessEqual:
            return "LessEqual";
        case DummyOpT::StridedSlice:
            return "StridedSlice";
        case DummyOpT::Pack:
            return "Pack";
        case DummyOpT::Unpack:
            return "Unpack";
        case DummyOpT::Reshape:
            return "Reshape";
        case DummyOpT::RealDiv:
            return "RealDiv";
        case DummyOpT::Sqrt:
            return "Sqrt";
        case DummyOpT::TruncatedNormal:
            return "TruncatedNormal";
        case DummyOpT::VariableV2:
            return "VariableV2";
        case DummyOpT::Assign:
            return "Assign";
        case DummyOpT::Identity:
            return "Identity";
        case DummyOpT::MatMul:
            return "MatMul";
        case DummyOpT::Transpose:
            return "Transpose";
        case DummyOpT::BatchMatMulV2:
            return "BatchMatMulV2";
        case DummyOpT::Sigmoid:
            return "Sigmoid";
        case DummyOpT::IdentityN:
            return "IdentityN";
        case DummyOpT::VarHandleOp:
            return "VarHandleOp";
        case DummyOpT::VarIsInitializedOp:
            return "VarIsInitializedOp";
        case DummyOpT::AssignVariableOp:
            return "AssignVariableOp";
        case DummyOpT::ReadVariableOp:
            return "ReadVariableOp";
        case DummyOpT::Mean:
            return "Mean";
        case DummyOpT::StopGradient:
            return "StopGradient";
        case DummyOpT::SquaredDifference:
            return "SquaredDifference";
        case DummyOpT::Rsqrt:
            return "Rsqrt";
        case DummyOpT::Sub:
            return "Sub";
        case DummyOpT::BiasAdd:
            return "BiasAdd";
        case DummyOpT::ConcatV2:
            return "ConcatV2";
        case DummyOpT::FloorDiv:
            return "FloorDiv";
        case DummyOpT::Tile:
            return "Tile";
        case DummyOpT::Neg:
            return "Neg";
        case DummyOpT::Relu:
            return "Relu";
        case DummyOpT::Abs:
            return "Abs";
        case DummyOpT::Squeeze:
            return "Squeeze";
        case DummyOpT::Greater:
            return "Greater";
        case DummyOpT::GreaterEqual:
            return "GreaterEqual";
        case DummyOpT::Select:
            return "Select";
        case DummyOpT::Exp:
            return "Exp";
        case DummyOpT::Log1p:
            return "Log1p";
        case DummyOpT::Equal:
            return "Equal";
        case DummyOpT::Switch:
            return "Switch";
        case DummyOpT::Merge:
            return "Merge";
        case DummyOpT::Fill:
            return "Fill";
        case DummyOpT::NoOp:
            return "NoOp";
        case DummyOpT::BroadcastGradientArgs:
            return "BroadcastGradientArgs";
        case DummyOpT::Sum:
            return "Sum";
        case DummyOpT::Shape:
            return "Shape";
        case DummyOpT::Reciprocal:
            return "Reciprocal";
        case DummyOpT::AddN:
            return "AddN";
        case DummyOpT::BiasAddGrad:
            return "BiasAddGrad";
        case DummyOpT::ReluGrad:
            return "ReluGrad";
        case DummyOpT::Sign:
            return "Sign";
        case DummyOpT::RsqrtGrad:
            return "RsqrtGrad";
        case DummyOpT::FloorMod:
            return "FloorMod";
        case DummyOpT::ConcatOffset:
            return "ConcatOffset";
        case DummyOpT::Slice:
            return "Slice";
        case DummyOpT::StridedSliceGrad:
            return "StridedSliceGrad";
        case DummyOpT::ZerosLike:
            return "ZerosLike";
        case DummyOpT::InvertPermutation:
            return "InvertPermutation";
        case DummyOpT::ApplyAdam:
            return "ApplyAdam";
        case DummyOpT::ResourceApplyAdam:
            return "ResourceApplyAdam";
        case DummyOpT::Snapshot:
            return "Snapshot";
        case DummyOpT::LogicalOr:
            return "LogicalOr";
        case DummyOpT::LogicalNot:
            return "LogicalNot";
        case DummyOpT::Prod:
            return "Prod";
        case DummyOpT::Where:
            return "Where";
        case DummyOpT::GatherV2:
            return "GatherV2";
        case DummyOpT::Split:
            return "Split";
        case DummyOpT::SplitV:
            return "SplitV";
        case DummyOpT::OneHot:
            return "OneHot";
        case DummyOpT::Pow:
            return "Pow";
        case DummyOpT::Softmax:
            return "Softmax";
        case DummyOpT::Tanh:
            return "Tanh";
        case DummyOpT::Pad:
            return "Pad";
        default:
            return "UnknownOp";
    }
}

template <DummyOpT dummyOp>
class NpuDummyOp : public OpKernel {
public:
    explicit NpuDummyOp(OpKernelConstruction* ctx) : OpKernel(ctx), name_(DummyOpTToString(dummyOp))
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

#define REGISTER_DUMMY_OP(op) REGISTER_KERNEL_BUILDER(Name(#op).Device(DEVICE_NPU), NpuDummyOp<DummyOpT::op>);

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