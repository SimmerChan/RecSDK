#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import math
import sysconfig

import pytest
import torch
import torch_npu

torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

DEVICE = "npu:7"
NUM_BUCKETS = 128


def create_rab_time_grad(num_layers: int, batchsize: int, s: int):
    nearest10 = math.ceil(math.log10(s+1))
    table = torch.arange(s * s).reshape(s, s) / (10 **nearest10)  # 用于排查哪些索引上的结果有问题
    batch = torch.arange(batchsize).reshape(-1, 1, 1)
    result = batch + table.unsqueeze(0)  # (b, s, s)
    return result.unsqueeze(0).repeat(num_layers, 1, 1, 1)  # (n, b, s, s)


def create_bucket_timestamps(batchsize: int, s: int):
    repeat_times = batchsize * s * s // NUM_BUCKETS + 1
    result = torch.arange(NUM_BUCKETS).repeat(repeat_times)[:batchsize * s * s]
    return result.reshape(batchsize, s, s)  # (b, s, s)


def rab_backward_golden(rab_time_grad: torch.Tensor, bucket_timestamps: torch.Tensor):
    num_layers, b, s, _ = rab_time_grad.shape
    tsw_grad = torch.zeros(num_layers, NUM_BUCKETS).to(rab_time_grad.device)

    bucket_timestamps_expand = (bucket_timestamps.reshape(b, s // 2, 1, s // 2, 1)
                                                 .repeat(1, 1, 2, 1, 2)
                                                 .reshape(b, s, s))
    for n, grad in enumerate(rab_time_grad):
        tsw_grad[n] = tsw_grad[n].scatter_add(src=grad.view(-1), index=bucket_timestamps_expand.view(-1), dim=0)
    return tsw_grad


def rab_backward_op(rab_time_grad: torch.Tensor, bucket_timestamps: torch.Tensor):
    return torch.ops.mxrec.relative_attn_bias_backward(rab_time_grad, bucket_timestamps, NUM_BUCKETS)


@torch.no_grad()
def rab_backward(num_layers: int, batchsize: int, s: int):
    grad = create_rab_time_grad(num_layers, batchsize, s)
    bucket_timestamps = create_bucket_timestamps(batchsize, s // 2)

    op_result = rab_backward_op(grad, bucket_timestamps)
    golden_result = rab_backward_golden(grad, bucket_timestamps)
    assert torch.allclose(op_result, golden_result)


if __name__ == '__main__':
    rab_backward(8, 1, 10)
