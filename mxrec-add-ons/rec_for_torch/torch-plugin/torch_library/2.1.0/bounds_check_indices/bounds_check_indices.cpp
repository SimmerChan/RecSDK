/**
 * @file bounds_check_indices.cpp
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

void boundsCheckIndicesNpu(at::Tensor& rows_per_table,
                           at::Tensor& indices,
                           at::Tensor& offsets,
                           int64_t boundsCheckMode,
                           at::Tensor& warning,
                           const c10::optional<at::Tensor>& weights,
                           const c10::optional<at::Tensor>& bOffsets,
                           const int64_t maxB)
{
    at::Tensor warningN = at::zeros({100}, warning.options());
    EXEC_NPU_CMD(aclnnBoundsCheckIndices, rows_per_table, indices, offsets, boundsCheckMode, warningN);
    warning[0] = warningN.sum();
}

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("bounds_check_indices(Tensor rows_per_table, \
                                Tensor indices, \
                                Tensor offsets, \
                                int boundsCheckMode, \
                                Tensor warning, \
                                Tensor? weights, \
                                Tensor? bOffsets, \
                                int maxB) -> ()");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("bounds_check_indices", &boundsCheckIndicesNpu);
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("bounds_check_indices", &boundsCheckIndicesNpu);
}
