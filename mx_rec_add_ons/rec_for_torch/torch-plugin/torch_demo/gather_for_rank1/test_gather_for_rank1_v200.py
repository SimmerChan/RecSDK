#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.
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
import numpy as np

torch.npu.config.allow_internal_format = False
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")


def get_loss(index: np.ndarray, weight: np.ndarray):
    weight_tensor = torch.nn.Parameter(torch.from_numpy(weight)).to(torch.float64)
    weight_tensor.retain_grad()

    index_tensor = torch.from_numpy(index).to(torch.int32)

    result = torch.index_select(weight_tensor, dim=0, index=index_tensor.view(-1))

    return result.cpu().detach().numpy()


def get_loss_op(index: np.ndarray, weight: np.ndarray, device: str):
    torch.npu.set_device(device)

    weight_tensor = torch.nn.Parameter(torch.from_numpy(weight)).to(device)
    weight_tensor.retain_grad()

    index_tensor = torch.from_numpy(index).to(device)

    result = torch.ops.mxrec.gather_for_rank1(weight_tensor, index_tensor.view(-1))

    return result.cpu().detach().numpy()


@pytest.mark.parametrize("embedding_dim", [10, 100, 1000])
@pytest.mark.parametrize("device", ["npu:0"])
def test_gather_for_rank1(embedding_dim, device):
    weight = np.random.randn(embedding_dim).astype(dtype=np.float32)

    index_shape = np.random.randint(9, size=3).astype(np.int32)
    index = np.random.randint(embedding_dim, size=np.power(2, index_shape)).astype(
        np.int32
    )

    result = get_loss(index, weight)
    result_op = get_loss_op(index, weight, device)

    assert np.allclose(result, result_op, atol=1e-6)
