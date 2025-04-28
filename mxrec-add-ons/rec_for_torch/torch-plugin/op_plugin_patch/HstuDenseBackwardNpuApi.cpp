/* Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.

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

#include <ATen/ops/zeros.h>

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

const size_t MIN_SEQ_LEN = 256;
const size_t MAX_SEQ_LEN = 4096;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> hstu_dense_backward(
    const at::Tensor& grad, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& attn_bias,
    const int64_t causal, const double silu_scale)
{
    TORCH_CHECK(q.dim() == 4, "The q should be 4D", OPS_ERROR(ErrCode::PARAM));  // 检查输入张量输入维度是否为4
    TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kFloat || q.scalar_type() == at::kBFloat16,
                "float16, float32 or bfloat16 tensor expected but got a tensor with dtype: ", q.scalar_type(),
                OPS_ERROR(ErrCode::PARAM));

    auto dense_grad = grad.contiguous();
    auto dense_q = q.contiguous();
    auto dense_k = k.contiguous();
    auto dense_v = v.contiguous();
    auto dense_attn_bias = attn_bias.contiguous();

    size_t batch_size = q.size(0);
    size_t seq = q.size(1);
    size_t num_heads = q.size(2);
    size_t block_nums = q.size(3);

    TORCH_CHECK(seq >= MIN_SEQ_LEN && seq <= MAX_SEQ_LEN, "seq_len expect in [256, 4096], but value is ", seq,
                OPS_ERROR(ErrCode::PARAM));

    at::Tensor grad_q = at::empty({batch_size, seq, num_heads, block_nums}, q.options());
    at::Tensor grad_k = at::empty({batch_size, seq, num_heads, block_nums}, q.options());
    at::Tensor grad_v = at::empty({batch_size, seq, num_heads, block_nums}, q.options());
    at::Tensor grad_attn_bias = at::zeros({batch_size, num_heads, seq, seq}, q.options());

    EXEC_NPU_CMD(aclnnHstuDenseBackward, dense_grad, dense_q, dense_k, dense_v, dense_attn_bias, causal, silu_scale,
                 grad_q, grad_k, grad_v, grad_attn_bias);

    return std::make_tuple(grad_q, grad_k, grad_v, grad_attn_bias);
}
}  // namespace op_api
