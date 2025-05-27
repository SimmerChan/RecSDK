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

import pytest
import numpy as np
import torch

torch.ops.load_library("./build/libsub_mul_concat.so")


def fused_op(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.ops.acl_ops.sub_mul_concat(x, y)


def torch_op(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.cat((x, y, x - y, x * y), dim=2)


@pytest
def test_fused_op():
    x = torch.randn((128, 10, 64), device="npu")
    y = torch.randn((128, 10, 64), device="npu")
    out = fused_op(x, y)
    gt = torch_op(x, y)
    assert np.allclose(out, gt)
