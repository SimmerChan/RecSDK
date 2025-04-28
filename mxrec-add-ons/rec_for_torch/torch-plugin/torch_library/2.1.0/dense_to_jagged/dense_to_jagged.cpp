/**
 * @file dense_to_jagged.cpp
 *
 * Copyright (C) 2024-2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "../common/pytorch_npu_helper.hpp"
using tensor_list = std::vector<at::Tensor>;
using namespace at;

at::Tensor dense_to_jagged_forward_npu(const at::Tensor& dense,
                                       const std::vector<at::Tensor>& offsets,
                                       const c10::optional<c10::SymInt>& total_L)
{
    const at::OptionalDeviceGuard guard(device_of(dense));
    auto D = dense.size(-1);
    auto dense_contin = dense.contiguous();

    int64_t total_L_computed;
    if (total_L.has_value()) {
        total_L_computed = total_L.value().expect_int();
    } else {
        total_L_computed = (int64_t)offsets.back().max().item<int64_t>();
    }

    auto output = at::empty({total_L_computed, D}, dense.options());
    EXEC_NPU_CMD(aclnnDenseToJagged, dense_contin, offsets[0], total_L_computed, output);
    return output;
};

std::tuple<at::Tensor, std::vector<at::Tensor>> dense_to_jagged_npu(const at::Tensor& dense,
                                                                    const std::vector<at::Tensor>& offsets,
                                                                    const c10::optional<int64_t>& total_L)
{
    int64_t total_L_computed;
    if (total_L.has_value()) {
        total_L_computed = total_L.value();
    } else {
        total_L_computed = (int64_t)offsets.back().max().item<int64_t>();
    }

    return {dense_to_jagged_forward_npu(dense, offsets, c10::SymInt(total_L_computed)), offsets};
};

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("dense_to_jagged_forward(Tensor dense, Tensor[] offsets, SymInt? total_L) -> Tensor");
    m.def("dense_to_jagged(Tensor dense, Tensor[] offsets, int? total_L) -> (Tensor, Tensor[])");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("dense_to_jagged_forward", &dense_to_jagged_forward_npu);
    m.impl("dense_to_jagged", &dense_to_jagged_npu);
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("dense_to_jagged_forward", &dense_to_jagged_forward_npu);
    m.impl("dense_to_jagged", &dense_to_jagged_npu);
}
