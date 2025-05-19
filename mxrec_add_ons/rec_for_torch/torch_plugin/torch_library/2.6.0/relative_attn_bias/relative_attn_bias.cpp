/**
 * @file relative_attn_bias.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <string>
#include <algorithm>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "../common/pytorch_npu_helper.hpp"
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using tensor_list = std::vector<at::Tensor>;
using namespace at;
using namespace std;

std::tuple<Tensor, Tensor> relative_attn_bias_impl_npu(
    const Tensor &rel_pos_bias,
    const Tensor &identity,
    const Tensor &timestamps,
    const Tensor &timestamps_weights,
    const at::IntArrayRef past_valid_lens,
    const double bucket_divisor)
{
    auto rel_pos_bias_conti = rel_pos_bias.contiguous();
    auto identity_conti = identity.contiguous();
    auto timestamps_conti = timestamps.contiguous();
    auto timestamps_weights_conti = timestamps_weights.contiguous();

    const int bs = past_valid_lens.size();
    const int s = rel_pos_bias.size(0);  // (2s, 2s)
    const int _s = s / 2;  // (2s, 2s)
    const int num_layers = timestamps_weights.size(0);

    at::Tensor rab_pos_out = at::zeros({bs, s, s}, rel_pos_bias_conti.options());
    at::Tensor rab_time_out = at::zeros({num_layers, bs, _s, 1, _s, 1}, timestamps_weights_conti.options());

    EXEC_NPU_CMD(aclnnRelativeAttnBias,
                 rel_pos_bias_conti,
                 identity_conti,
                 timestamps_conti,
                 timestamps_weights_conti,
                 past_valid_lens,
                 bucket_divisor,
                 rab_pos_out,
                 rab_time_out);
    rab_time_out = rab_time_out.repeat({1, 1, 1, 2, 1, 2})
                               .reshape({num_layers, bs, s, s});
    return {rab_pos_out, rab_time_out};
}

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("relative_attn_bias(Tensor rel_pos_bias, "
          "                   Tensor identity, "
          "                   Tensor timestamps, "
          "                   Tensor timestamps_weights, "
          "                   int[] past_valid_lens,"
          "                   float bucket_divisor"
          "                   ) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("relative_attn_bias", &relative_attn_bias_impl_npu);
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("relative_attn_bias", &relative_attn_bias_impl_npu);
}
