/**
 * @file expand_into_jagged_permute.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "../common/pytorch_npu_helper.hpp"
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::Variable;
using tensor_list = std::vector<at::Tensor>;
using namespace at;

constexpr int EXPECTED_DIM_1D = 1;

void validate_expand_into_jagged_permute_inputs(
    const at::Tensor& permute,
    const at::Tensor& input_offset,
    const at::Tensor& output_offsets,
    const int64_t output_size)
{
    // ============= 空值检查 =============
    check_tensor_non_empty(permute, "permute");
    check_tensor_non_empty(input_offset, "input_offset");
    check_tensor_non_empty(output_offsets, "output_offsets");

    check_tensor_dim(permute, EXPECTED_DIM_1D, "permute");
    check_tensor_dim(input_offset, EXPECTED_DIM_1D, "input_offset");
    check_tensor_dim(output_offsets, EXPECTED_DIM_1D, "output_offsets");

    const auto permute_len = permute.size(0);
    const auto input_offset_len = input_offset.size(0);
    const auto output_offsets_len = output_offsets.size(0);

    // 1. 校验inputOffset和outputOffset的shape要相同
    TORCH_CHECK(input_offset_len == output_offsets_len,
                "input_offset_len and output_offsets_len must be the same, but got input_offset_len: ",
                input_offset_len, " and output_offsets_len: ", output_offsets_len);

    TORCH_CHECK(permute_len == input_offset_len - 1,
                "permute_len must equals input_offset_len - 1, but got permute_len: ",
                permute_len, " and input_offset_len: ", input_offset_len);

    // 2. 校验所有输入张量的数据类型相同
    TORCH_CHECK(permute.scalar_type() == input_offset.scalar_type(),
                "permute and input_offset must have the same data type, but got permute: ",
                permute.scalar_type(), " and input_offset: ", input_offset.scalar_type());

    TORCH_CHECK(permute.scalar_type() == output_offsets.scalar_type(),
                "permute and output_offsets must have the same data type, but got permute: ",
                permute.scalar_type(), " and output_offsets: ", output_offsets.scalar_type());

    // 3. 校验outputOffset最后一个值等于output_size
    if (output_offsets.numel() > 0) {
        auto last_offset = output_offsets[-1].item<int64_t>();
        TORCH_CHECK(last_offset == output_size,
                    "Last value of output_offsets (", last_offset,
                    ") must equal output_size (", output_size, ")");
    } else {
        TORCH_CHECK(output_size == 0,
                    "output_size must be 0 when output_offsets is empty, but got ", output_size);
    }
}

at::Tensor expand_into_jagged_permute_impl_npu(const at::Tensor& permute,
                                               const at::Tensor& input_offset,
                                               const at::Tensor& output_offsets,
                                               const int64_t output_size)
{
    validate_expand_into_jagged_permute_inputs(permute,
                                               input_offset,
                                               output_offsets,
                                               output_size);

    const at::OptionalDeviceGuard guard(device_of(permute));
    at::Tensor outputPermuteOut = at::empty({output_size}, permute.options());

    EXEC_NPU_CMD(aclnnExpandIntoJaggedPermute, permute, input_offset, output_offsets, output_size, outputPermuteOut);

    return outputPermuteOut;
};

// 在NPU命名空间里面注册expand_into_jagged_permute
TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("expand_into_jagged_permute(Tensor permute, "
          "                           Tensor input_offset, "
          "                           Tensor output_offsets, "
          "                           int output_size) -> Tensor");
}

// 这里表示该算子的 NPU 实现由 expand_into_jagged_permute_impl_npu 函数提供
TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("expand_into_jagged_permute", &expand_into_jagged_permute_impl_npu);
}

// 将同一个算子同时注册到 fbgemm 库的 PrivateUse1 后端
TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("expand_into_jagged_permute", &expand_into_jagged_permute_impl_npu);
}