/**
 * @file FbgemmNpuApi.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
 
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch/extension.h"

#define DISPATCH_TO_NPU(name, function) m.impl(name, torch::dispatch(c10::DispatchKey::PrivateUse1, TORCH_FN(function)))

at::Tensor dense_to_jagged_forward_npu(const at::Tensor& dense, const std::vector<at::Tensor>& offsets,
                                       c10::optional<at::SymInt> total_L)
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

at::Tensor asynchronous_complete_cumsum_npu(const at::Tensor& offset)
{
    const at::OptionalDeviceGuard guard(device_of(offset));
    auto offset_contin = offset.contiguous();
    auto output = at::empty({offset.size(0) + 1}, offset.options());

    EXEC_NPU_CMD(aclnnAsynchronousCompleteCumsum, offset_contin, output);
    return output;
};

at::Tensor jagged_to_padded_dense_forward_npu(const at::Tensor& values, const std::vector<at::Tensor>& offsets,
                                              c10::SymIntArrayRef max_lengths, const double padding_value)
{
    const at::OptionalDeviceGuard guard(device_of(values));
    auto values_contin = values.contiguous();
    auto D = values.size(-1);
    at::IntArrayRef max_lengths_int = c10::asIntArrayRefUnchecked(max_lengths);
    auto output =
        at::full({offsets[0].size(0) - 1, max_lengths_int.at(0), values.size(1)}, padding_value, values.options());
    EXEC_NPU_CMD(aclnnJaggedToPaddedDense, values_contin, offsets[0], max_lengths_int.at(0), padding_value, output);
    return output;
};

at::Tensor jagged_to_padded_dense_backward_npu(const at::Tensor& grad_output, const std::vector<at::Tensor>& offsets,
                                               at::SymInt total_L)
{
    return dense_to_jagged_forward_npu(grad_output, offsets, total_L);
};

at::Tensor jagged_to_padded_dense_npu(const at::Tensor& values, const std::vector<at::Tensor>& offsets,
                                      c10::SymIntArrayRef max_lengths, const double padding_value)
{
    return jagged_to_padded_dense_forward_npu(values, offsets, max_lengths, padding_value);
};

std::tuple<at::Tensor, std::vector<at::Tensor>> dense_to_jagged_npu(const at::Tensor& dense,
                                                                            const std::vector<at::Tensor>& offsets,
                                                                            c10::optional<int64_t> total_L)
{
    int64_t total_L_computed;
    if (total_L.has_value()) {
        total_L_computed = total_L.value();
    } else {
        total_L_computed = (int64_t)offsets.back().max().item<int64_t>();
    }

    return {dense_to_jagged_forward_npu(dense, offsets, at::SymInt(total_L_computed)), offsets};
};

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    DISPATCH_TO_NPU("dense_to_jagged_forward", dense_to_jagged_forward_npu);
    DISPATCH_TO_NPU("asynchronous_complete_cumsum", asynchronous_complete_cumsum_npu);
    DISPATCH_TO_NPU("jagged_to_padded_dense_forward", jagged_to_padded_dense_forward_npu);
    DISPATCH_TO_NPU("jagged_to_padded_dense_backward", jagged_to_padded_dense_backward_npu);
    DISPATCH_TO_NPU("jagged_to_padded_dense", jagged_to_padded_dense_npu);
    DISPATCH_TO_NPU("dense_to_jagged", dense_to_jagged_npu);
};
