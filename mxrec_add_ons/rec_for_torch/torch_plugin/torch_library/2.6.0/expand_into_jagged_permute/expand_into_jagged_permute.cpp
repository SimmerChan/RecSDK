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

void check_tensor_not_empty(const Tensor &tensor, const std::string &name)
{
    TORCH_CHECK(tensor.defined(), name, " tensor must be defined");
    TORCH_CHECK(tensor.numel() > 0, name, " tensor must be non-empty");
}

void validate_expand_into_jagged_permute_inputs(
    const at::Tensor& permute,
    const at::Tensor& input_offset,
    const at::Tensor& output_offsets,
    const int32_t output_size)
{
    // ============= 空值检查 =============
    check_tensor_not_empty(permute, "permute");
    check_tensor_not_empty(input_offset, "input_offset");
    check_tensor_not_empty(output_offsets, "output_offsets");

    // 1. 校验inputOffset和outputOffset的shape要相同
    TORCH_CHECK(input_offset.sizes().equals(output_offsets.sizes()),
                "input_offset and output_offsets must have the same shape, but got input_offset: ",
                input_offset.sizes(), " and output_offsets: ", output_offsets.sizes());

    // 2. 校验所有输入张量的数据类型相同
    TORCH_CHECK(permute.scalar_type() == input_offset.scalar_type(),
                "permute and input_offset must have the same data type, but got permute: ",
                permute.scalar_type(), " and input_offset: ", input_offset.scalar_type());

    TORCH_CHECK(permute.scalar_type() == output_offsets.scalar_type(),
                "permute and output_offsets must have the same data type, but got permute: ",
                permute.scalar_type(), " and output_offsets: ", output_offsets.scalar_type());

    // 3. 校验outputOffset的值严格单调递增
    if (output_offsets.numel() > 1) {
        auto output_offsets_accessor = output_offsets.accessor<int32_t, 1>();
        for (int32_t i = 1; i < output_offsets.numel(); ++i) {
            TORCH_CHECK(output_offsets_accessor[i] > output_offsets_accessor[i-1],
                        "output_offsets must be strictly monotonically increasing, but found ",
                        output_offsets_accessor[i-1], " >= ", output_offsets_accessor[i], " at index ", i);
        }
    }

    // 4. 校验outputOffset最后一个值等于output_size
    if (output_offsets.numel() > 0) {
        auto last_offset = output_offsets[-1].item<int32_t>();
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
                                               const int32_t output_size)
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