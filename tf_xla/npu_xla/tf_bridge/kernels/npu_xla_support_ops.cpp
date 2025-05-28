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
    CONST = 0,
    RANDOM_STANDARD_NORMAL,
    MUL,
    ADD_V2,
    RANDOM_UNIFORM_INT,
    RANDOM_UNIFORM,
    MAX,
    MAXIMUM,
    MINIMUM,
    RANGE,
    EXPAND_DIMS,
    CAST,
    LESS,
    LESS_EQUAL,
    STRIDED_SLICE,
    PACK,
    UNPACK,
    RESHAPE,
    REAL_DIV,
    SQRT,
    TRUNCATED_NORMAL,
    VARIABLE_V2,
    ASSIGN,
    IDENTITY,
    MAT_MUL,
    TRANSPOSE,
    BATCH_MAT_MUL_V2,
    SIGMOID,
    IDENTITY_N,
    VAR_HANDLE_OP,
    VAR_IS_INITIALIZED_OP,
    ASSIGN_VARIABLE_OP,
    READ_VARIABLE_OP,
    MEAN,
    STOP_GRADIENT,
    SQUARED_DIFFERENCE,
    RSQRT,
    SUB,
    BIAS_ADD,
    CONCAT_V2,
    FLOOR_DIV,
    TILE,
    NEG,
    RELU,
    ABS,
    SQUEEZE,
    GREATER,
    GREATER_EQUAL,
    SELECT,
    EXP,
    LOG1P,
    EQUAL,
    SWITCH,
    MERGE,
    FILL,
    NO_OP,
    BROADCAST_GRADIENT_ARGS,
    SUM,
    SHAPE,
    RECIPROCAL,
    ADD_N,
    BIAS_ADD_GRAD,
    RELU_GRAD,
    SIGN,
    RSQRT_GRAD,
    FLOOR_MOD,
    CONCAT_OFFSET,
    SLICE,
    STRIDED_SLICE_GRAD,
    ZEROS_LIKE,
    INVERT_PERMUTATION,
    APPLY_ADAM,
    RESOURCE_APPLY_ADAM,
    SNAPSHOT,
    LOGICAL_OR,
    LOGICAL_NOT,
    PROD,
    WHERE,
    GATHER_V2,
    SPLIT,
    SPLIT_V,
    ONE_HOT,
    POW,
    SOFTMAX,
    TANH,
    PAD,
};

std::string DummyOpTToString(DummyOpT op)
{
    switch (op) {
        case DummyOpT::CONST:
            return "Const";
        case DummyOpT::RANDOM_STANDARD_NORMAL:
            return "RandomStandardNormal";
        case DummyOpT::MUL:
            return "Mul";
        case DummyOpT::ADD_V2:
            return "AddV2";
        case DummyOpT::RANDOM_UNIFORM_INT:
            return "RandomUniformInt";
        case DummyOpT::RANDOM_UNIFORM:
            return "RandomUniform";
        case DummyOpT::MAX:
            return "Max";
        case DummyOpT::MAXIMUM:
            return "Maximum";
        case DummyOpT::MINIMUM:
            return "Minimum";
        case DummyOpT::RANGE:
            return "Range";
        case DummyOpT::EXPAND_DIMS:
            return "ExpandDims";
        case DummyOpT::CAST:
            return "Cast";
        case DummyOpT::LESS:
            return "Less";
        case DummyOpT::LESS_EQUAL:
            return "LessEqual";
        case DummyOpT::STRIDED_SLICE:
            return "StridedSlice";
        case DummyOpT::PACK:
            return "Pack";
        case DummyOpT::UNPACK:
            return "Unpack";
        case DummyOpT::RESHAPE:
            return "Reshape";
        case DummyOpT::REAL_DIV:
            return "RealDiv";
        case DummyOpT::SQRT:
            return "Sqrt";
        case DummyOpT::TRUNCATED_NORMAL:
            return "TruncatedNormal";
        case DummyOpT::VARIABLE_V2:
            return "VariableV2";
        case DummyOpT::ASSIGN:
            return "Assign";
        case DummyOpT::IDENTITY:
            return "Identity";
        case DummyOpT::MAT_MUL:
            return "MatMul";
        case DummyOpT::TRANSPOSE:
            return "Transpose";
        case DummyOpT::BATCH_MAT_MUL_V2:
            return "BatchMatMulV2";
        case DummyOpT::SIGMOID:
            return "Sigmoid";
        case DummyOpT::IDENTITY_N:
            return "IdentityN";
        case DummyOpT::VAR_HANDLE_OP:
            return "VarHandleOp";
        case DummyOpT::VAR_IS_INITIALIZED_OP:
            return "VarIsInitializedOp";
        case DummyOpT::ASSIGN_VARIABLE_OP:
            return "AssignVariableOp";
        case DummyOpT::READ_VARIABLE_OP:
            return "ReadVariableOp";
        case DummyOpT::MEAN:
            return "Mean";
        case DummyOpT::STOP_GRADIENT:
            return "StopGradient";
        case DummyOpT::SQUARED_DIFFERENCE:
            return "SquaredDifference";
        case DummyOpT::RSQRT:
            return "Rsqrt";
        case DummyOpT::SUB:
            return "Sub";
        case DummyOpT::BIAS_ADD:
            return "BiasAdd";
        case DummyOpT::CONCAT_V2:
            return "ConcatV2";
        case DummyOpT::FLOOR_DIV:
            return "FloorDiv";
        case DummyOpT::TILE:
            return "Tile";
        case DummyOpT::NEG:
            return "Neg";
        case DummyOpT::RELU:
            return "Relu";
        case DummyOpT::ABS:
            return "Abs";
        case DummyOpT::SQUEEZE:
            return "Squeeze";
        case DummyOpT::GREATER:
            return "Greater";
        case DummyOpT::GREATER_EQUAL:
            return "GreaterEqual";
        case DummyOpT::SELECT:
            return "Select";
        case DummyOpT::EXP:
            return "Exp";
        case DummyOpT::LOG1P:
            return "Log1p";
        case DummyOpT::EQUAL:
            return "Equal";
        case DummyOpT::SWITCH:
            return "Switch";
        case DummyOpT::MERGE:
            return "Merge";
        case DummyOpT::FILL:
            return "Fill";
        case DummyOpT::NO_OP:
            return "NoOp";
        case DummyOpT::BROADCAST_GRADIENT_ARGS:
            return "BroadcastGradientArgs";
        case DummyOpT::SUM:
            return "Sum";
        case DummyOpT::SHAPE:
            return "Shape";
        case DummyOpT::RECIPROCAL:
            return "Reciprocal";
        case DummyOpT::ADD_N:
            return "AddN";
        case DummyOpT::BIAS_ADD_GRAD:
            return "BiasAddGrad";
        case DummyOpT::RELU_GRAD:
            return "ReluGrad";
        case DummyOpT::SIGN:
            return "Sign";
        case DummyOpT::RSQRT_GRAD:
            return "RsqrtGrad";
        case DummyOpT::FLOOR_MOD:
            return "FloorMod";
        case DummyOpT::CONCAT_OFFSET:
            return "ConcatOffset";
        case DummyOpT::SLICE:
            return "Slice";
        case DummyOpT::STRIDED_SLICE_GRAD:
            return "StridedSliceGrad";
        case DummyOpT::ZEROS_LIKE:
            return "ZerosLike";
        case DummyOpT::INVERT_PERMUTATION:
            return "InvertPermutation";
        case DummyOpT::APPLY_ADAM:
            return "ApplyAdam";
        case DummyOpT::RESOURCE_APPLY_ADAM:
            return "ResourceApplyAdam";
        case DummyOpT::SNAPSHOT:
            return "Snapshot";
        case DummyOpT::LOGICAL_OR:
            return "LogicalOr";
        case DummyOpT::LOGICAL_NOT:
            return "LogicalNot";
        case DummyOpT::PROD:
            return "Prod";
        case DummyOpT::WHERE:
            return "Where";
        case DummyOpT::GATHER_V2:
            return "GatherV2";
        case DummyOpT::SPLIT:
            return "Split";
        case DummyOpT::SPLIT_V:
            return "SplitV";
        case DummyOpT::ONE_HOT:
            return "OneHot";
        case DummyOpT::POW:
            return "Pow";
        case DummyOpT::SOFTMAX:
            return "Softmax";
        case DummyOpT::TANH:
            return "Tanh";
        case DummyOpT::PAD:
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

#define REGISTER_DUMMY_OP(enumName, stringName) \
    REGISTER_KERNEL_BUILDER(Name(stringName).Device(DEVICE_NPU), NpuDummyOp<DummyOpT::enumName>);

REGISTER_DUMMY_OP(ADD_V2, "AddV2");
REGISTER_DUMMY_OP(CAST, "Cast");
REGISTER_DUMMY_OP(MAXIMUM, "Maximum");
REGISTER_DUMMY_OP(MINIMUM, "Minimum");
REGISTER_DUMMY_OP(RANDOM_STANDARD_NORMAL, "RandomStandardNormal");
REGISTER_DUMMY_OP(MUL, "Mul");
REGISTER_DUMMY_OP(RANDOM_UNIFORM_INT, "RandomUniformInt");
REGISTER_DUMMY_OP(RANDOM_UNIFORM, "RandomUniform");
REGISTER_DUMMY_OP(SUB, "Sub");
REGISTER_DUMMY_OP(RANGE, "Range");
REGISTER_DUMMY_OP(MAX, "Max");
REGISTER_DUMMY_OP(EXPAND_DIMS, "ExpandDims");
REGISTER_DUMMY_OP(LESS, "Less");
REGISTER_DUMMY_OP(LESS_EQUAL, "LessEqual");
REGISTER_DUMMY_OP(STRIDED_SLICE, "StridedSlice");
REGISTER_DUMMY_OP(PACK, "Pack");
REGISTER_DUMMY_OP(UNPACK, "Unpack");
REGISTER_DUMMY_OP(RESHAPE, "Reshape");
REGISTER_DUMMY_OP(REAL_DIV, "RealDiv");
REGISTER_DUMMY_OP(SQRT, "Sqrt");
REGISTER_DUMMY_OP(TRUNCATED_NORMAL, "TruncatedNormal");
REGISTER_DUMMY_OP(MAT_MUL, "MatMul");
REGISTER_DUMMY_OP(TRANSPOSE, "Transpose");
REGISTER_DUMMY_OP(BATCH_MAT_MUL_V2, "BatchMatMulV2");
REGISTER_DUMMY_OP(SIGMOID, "Sigmoid");
REGISTER_DUMMY_OP(IDENTITY_N, "IdentityN");
REGISTER_DUMMY_OP(ASSIGN_VARIABLE_OP, "AssignVariableOp");
REGISTER_DUMMY_OP(READ_VARIABLE_OP, "ReadVariableOp");
REGISTER_DUMMY_OP(MEAN, "Mean");
REGISTER_DUMMY_OP(STOP_GRADIENT, "StopGradient");
REGISTER_DUMMY_OP(SQUARED_DIFFERENCE, "SquaredDifference");
REGISTER_DUMMY_OP(RSQRT, "Rsqrt");
REGISTER_DUMMY_OP(BIAS_ADD, "BiasAdd");
REGISTER_DUMMY_OP(CONCAT_V2, "ConcatV2");
REGISTER_DUMMY_OP(FLOOR_DIV, "FloorDiv");
REGISTER_DUMMY_OP(TILE, "Tile");
REGISTER_DUMMY_OP(NEG, "Neg");
REGISTER_DUMMY_OP(RELU, "Relu");
REGISTER_DUMMY_OP(ABS, "Abs");
REGISTER_DUMMY_OP(SQUEEZE, "Squeeze");
REGISTER_DUMMY_OP(GREATER, "Greater");
REGISTER_DUMMY_OP(GREATER_EQUAL, "GreaterEqual");
REGISTER_DUMMY_OP(SELECT, "Select");
REGISTER_DUMMY_OP(EXP, "Exp");
REGISTER_DUMMY_OP(LOG1P, "Log1p");
REGISTER_DUMMY_OP(EQUAL, "Equal");
REGISTER_DUMMY_OP(MERGE, "Merge");
REGISTER_DUMMY_OP(FILL, "Fill");
REGISTER_DUMMY_OP(BROADCAST_GRADIENT_ARGS, "BroadcastGradientArgs");
REGISTER_DUMMY_OP(SUM, "Sum");
REGISTER_DUMMY_OP(SHAPE, "Shape");
REGISTER_DUMMY_OP(RECIPROCAL, "Reciprocal");
REGISTER_DUMMY_OP(ADD_N, "AddN");
REGISTER_DUMMY_OP(BIAS_ADD_GRAD, "BiasAddGrad");
REGISTER_DUMMY_OP(RELU_GRAD, "ReluGrad");
REGISTER_DUMMY_OP(SIGN, "Sign");
REGISTER_DUMMY_OP(RSQRT_GRAD, "RsqrtGrad");
REGISTER_DUMMY_OP(FLOOR_MOD, "FloorMod");
REGISTER_DUMMY_OP(CONCAT_OFFSET, "ConcatOffset");
REGISTER_DUMMY_OP(SLICE, "Slice");
REGISTER_DUMMY_OP(STRIDED_SLICE_GRAD, "StridedSliceGrad");
REGISTER_DUMMY_OP(ZEROS_LIKE, "ZerosLike");
REGISTER_DUMMY_OP(INVERT_PERMUTATION, "InvertPermutation");
REGISTER_DUMMY_OP(APPLY_ADAM, "ApplyAdam");
REGISTER_DUMMY_OP(RESOURCE_APPLY_ADAM, "ResourceApplyAdam");
REGISTER_DUMMY_OP(LOGICAL_OR, "LogicalOr");
REGISTER_DUMMY_OP(LOGICAL_NOT, "LogicalNot");
REGISTER_DUMMY_OP(PROD, "Prod");
REGISTER_DUMMY_OP(WHERE, "Where");
REGISTER_DUMMY_OP(GATHER_V2, "GatherV2");
REGISTER_DUMMY_OP(SPLIT, "Split");
REGISTER_DUMMY_OP(SPLIT_V, "SplitV");
REGISTER_DUMMY_OP(ONE_HOT, "OneHot");
REGISTER_DUMMY_OP(POW, "Pow");
REGISTER_DUMMY_OP(TANH, "Tanh");
REGISTER_DUMMY_OP(PAD, "Pad");
}  // namespace tensorflow