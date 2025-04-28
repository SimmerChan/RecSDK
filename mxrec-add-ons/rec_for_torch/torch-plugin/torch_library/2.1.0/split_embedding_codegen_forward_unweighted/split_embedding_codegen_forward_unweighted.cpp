/**
 * @file split_embedding_codegen_forward_unweighted.cpp
 *
 * Copyright (C) 2024-2025. Huawei Technologies Co., Ltd. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "torch/extension.h"
#include "../common/pytorch_npu_helper.hpp"
#include "split_embedding_codegen_forward_unweighted.h"
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;
using tensor_list = std::vector<at::Tensor>;
using Tensor = at::Tensor;
using namespace at;

// using namespace fbgemm_gpu;
namespace fbgemm_npu_lookups {
Tensor split_embedding_codegen_forward_unweighted_cuda(const Tensor& dev_weights,
                                                       const Tensor& uvm_weights,
                                                       const Tensor& lxu_cache_weights,
                                                       const Tensor& weights_placements,
                                                       const Tensor& weights_offsets,
                                                       const Tensor& D_offsets,
                                                       const int64_t total_D,
                                                       const int64_t max_D,
                                                       const Tensor& indices,
                                                       const Tensor& offsets,
                                                       const int64_t pooling_mode,
                                                       const Tensor& lxu_cache_locations,
                                                       const int64_t output_dtype,
                                                       const bool is_experimental,
                                                       const Tensor& hash_indices);

Tensor split_embedding_codegen_forward_unweighted_npu(const Tensor& dev_weights,
                                                      const Tensor& uvm_weights,
                                                      const Tensor& lxu_cache_weights,
                                                      const Tensor& weights_placements,
                                                      const Tensor& weights_offsets,
                                                      const Tensor& D_offsets,
                                                      const int64_t total_D,
                                                      const int64_t max_D,
                                                      const Tensor& indices,
                                                      const Tensor& offsets,
                                                      const int64_t pooling_mode,
                                                      const Tensor& lxu_cache_locations,
                                                      const int64_t output_dtype,
                                                      const bool is_experimental,
                                                      const Tensor& hash_indices)
{
    const at::OptionalDeviceGuard guard(device_of(dev_weights));

    int64_t featCnt = weights_placements.size(0);
    int32_t totalLen = indices.numel();
    if (featCnt == 0) {
        return Tensor();
    }

    if (totalLen == 0) {
        return Tensor();
    }

    int64_t batchSize = (offsets.size(0) - 1) / featCnt;
    auto output = at::full({batchSize, total_D}, 0.0, dev_weights.options());

    if (static_cast<PoolingMode>(pooling_mode) == PoolingMode::NONE) {
        output = at::full({totalLen, max_D}, 0.0, dev_weights.options());
    }

    EXEC_NPU_CMD(aclnnSplitEmbeddingCodegenForwardUnweighted, dev_weights, uvm_weights, lxu_cache_weights,
                 weights_placements, weights_offsets, D_offsets, indices, offsets, lxu_cache_locations, hash_indices,
                 total_D, max_D, pooling_mode, output_dtype, is_experimental, output);
    return output;
}
}; // namespace fbgemm_npu_lookups

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    // 注册正向传播
    DISPATCH_TO_NPU("split_embedding_codegen_forward_unweighted_cuda",
                    fbgemm_npu_lookups::split_embedding_codegen_forward_unweighted_npu);
}
