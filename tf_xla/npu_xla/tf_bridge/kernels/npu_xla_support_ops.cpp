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

namespace tensorflow {
const char* const DEVICE_NPU = "NPU";
enum DummyOpT {
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
        case Const:
            return "Const";
        case RandomStandardNormal:
            return "RandomStandardNormal";
        case Mul:
            return "Mul";
        case AddV2:
            return "AddV2";
        case RandomUniformInt:
            return "RandomUniformInt";
        case RandomUniform:
            return "RandomUniform";
        case Max:
            return "Max";
        case Maximum:
            return "Maximum";
        case Minimum:
            return "Minimum";
        case Range:
            return "Range";
        case ExpandDims:
            return "ExpandDims";
        case Cast:
            return "Cast";
        case Less:
            return "Less";
        case LessEqual:
            return "LessEqual";
        case StridedSlice:
            return "StridedSlice";
        case Pack:
            return "Pack";
        case Unpack:
            return "Unpack";
        case Reshape:
            return "Reshape";
        case RealDiv:
            return "RealDiv";
        case Sqrt:
            return "Sqrt";
        case TruncatedNormal:
            return "TruncatedNormal";
        case VariableV2:
            return "VariableV2";
        case Assign:
            return "Assign";
        case Identity:
            return "Identity";
        case MatMul:
            return "MatMul";
        case Transpose:
            return "Transpose";
        case BatchMatMulV2:
            return "BatchMatMulV2";
        case Sigmoid:
            return "Sigmoid";
        case IdentityN:
            return "IdentityN";
        case VarHandleOp:
            return "VarHandleOp";
        case VarIsInitializedOp:
            return "VarIsInitializedOp";
        case AssignVariableOp:
            return "AssignVariableOp";
        case ReadVariableOp:
            return "ReadVariableOp";
        case Mean:
            return "Mean";
        case StopGradient:
            return "StopGradient";
        case SquaredDifference:
            return "SquaredDifference";
        case Rsqrt:
            return "Rsqrt";
        case Sub:
            return "Sub";
        case BiasAdd:
            return "BiasAdd";
        case ConcatV2:
            return "ConcatV2";
        case FloorDiv:
            return "FloorDiv";
        case Tile:
            return "Tile";
        case Neg:
            return "Neg";
        case Relu:
            return "Relu";
        case Abs:
            return "Abs";
        case Squeeze:
            return "Squeeze";
        case Greater:
            return "Greater";
        case GreaterEqual:
            return "GreaterEqual";
        case Select:
            return "Select";
        case Exp:
            return "Exp";
        case Log1p:
            return "Log1p";
        case Equal:
            return "Equal";
        case Switch:
            return "Switch";
        case Merge:
            return "Merge";
        case Fill:
            return "Fill";
        case NoOp:
            return "NoOp";
        case BroadcastGradientArgs:
            return "BroadcastGradientArgs";
        case Sum:
            return "Sum";
        case Shape:
            return "Shape";
        case Reciprocal:
            return "Reciprocal";
        case AddN:
            return "AddN";
        case BiasAddGrad:
            return "BiasAddGrad";
        case ReluGrad:
            return "ReluGrad";
        case Sign:
            return "Sign";
        case RsqrtGrad:
            return "RsqrtGrad";
        case FloorMod:
            return "FloorMod";
        case ConcatOffset:
            return "ConcatOffset";
        case Slice:
            return "Slice";
        case StridedSliceGrad:
            return "StridedSliceGrad";
        case ZerosLike:
            return "ZerosLike";
        case InvertPermutation:
            return "InvertPermutation";
        case ApplyAdam:
            return "ApplyAdam";
        case ResourceApplyAdam:
            return "ResourceApplyAdam";
        case Snapshot:
            return "Snapshot";
        case LogicalOr:
            return "LogicalOr";
        case LogicalNot:
            return "LogicalNot";
        case Prod:
            return "Prod";
        case Where:
            return "Where";
        case GatherV2:
            return "GatherV2";
        case Split:
            return "Split";
        case SplitV:
            return "SplitV";
        case OneHot:
            return "OneHot";
        case Pow:
            return "Pow";
        case Tanh:
            return "Tanh";
        case Pad:
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
        VLOG(1) << "Construct NPU dummy op: " << name_;
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
// REGISTER_DUMMY_OP(Identity);
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
// REGISTER_DUMMY_OP(VariableV2);
// REGISTER_DUMMY_OP(Assign);
REGISTER_DUMMY_OP(MatMul);
REGISTER_DUMMY_OP(Transpose);
REGISTER_DUMMY_OP(BatchMatMulV2);
REGISTER_DUMMY_OP(Sigmoid);
REGISTER_DUMMY_OP(IdentityN);
// REGISTER_DUMMY_OP(VarHandleOp);
// REGISTER_DUMMY_OP(VarIsInitializedOp);
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
// REGISTER_DUMMY_OP(Switch);
REGISTER_DUMMY_OP(Merge);
REGISTER_DUMMY_OP(Fill);
// REGISTER_DUMMY_OP(NoOp);
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
// REGISTER_DUMMY_OP(Snapshot);
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