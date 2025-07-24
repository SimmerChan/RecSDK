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
import sysconfig

import pytest
import torch
import torch_npu
import fbgemm_gpu
import numpy as np

DEVICE = "npu:7"
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")


def get_result(permute, lengths, values, weights, permuted_lengths_sum, device: str = 'cpu'):
    tensors = {
        'permute': torch.from_numpy(permute),
        'lengths': torch.from_numpy(lengths),
        'values': torch.from_numpy(values),
        'weights': torch.from_numpy(weights) if isinstance(weights, torch.Tensor) else None
    }

    if device and device.startswith('npu'):
        torch.npu.set_device(device)
        tensors = {k: v.to(device) if isinstance(v, torch.Tensor) else None for k, v in tensors.items()}

    results = torch.ops.fbgemm.permute_2D_sparse_data(
        permuted_lengths_sum=permuted_lengths_sum, **tensors
    )
    return tuple(result.cpu() if isinstance(result, torch.Tensor) else result for result in results)


@pytest.mark.parametrize("ltype", [np.int64, np.int32])
@pytest.mark.parametrize("vtype", [np.int64, np.int32, np.float32])
@pytest.mark.parametrize("wtype", [None, np.float32])
@pytest.mark.parametrize("permute_dim", np.random.randint(2, 30, 4).tolist())
@pytest.mark.parametrize("extra_permute_dim", [0, 3, 8])
@pytest.mark.parametrize("permuted_lengths_sum", [True, False])
@pytest.mark.parametrize("lengths", [2048, 20480, 204800])
def test_permute2d_sparse_data(ltype, vtype, wtype, permute_dim, extra_permute_dim, permuted_lengths_sum, lengths):
    permute = np.arange(permute_dim, dtype=np.int32)
    np.random.shuffle(permute)
    values = np.arange(0, (permute_dim + extra_permute_dim) * lengths, dtype=vtype)
    weights = np.arange(0, (permute_dim + extra_permute_dim) * lengths, dtype=wtype) if wtype else None
    lengths = np.ones((permute_dim + extra_permute_dim, lengths), dtype=ltype)
    permuted_lengths_sum = lengths[:permute_dim].sum() if permuted_lengths_sum else None

    golden = get_result(permute, lengths, values, weights, permuted_lengths_sum)
    result = get_result(permute, lengths, values, weights, permuted_lengths_sum, DEVICE)

    for gt, pred in zip(golden, result):
        assert type(gt) is type(pred)
        if isinstance(gt, torch.Tensor) and isinstance(pred, torch.Tensor):
            assert torch.allclose(gt, pred, atol=1e-5)
