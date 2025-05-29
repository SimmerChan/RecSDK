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

import torch
from torch._inductor.pattern_matcher import (
    fwd_only,
    PatternMatcherPass,
    register_replacement,
)
import torch._inductor.config as inductor_config
import torch_npu._inductor


torch.ops.load_library("./torch_op_plugin/build/libsub_mul_concat.so")


def fused_sub_mul_concat(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    print("fused_sub_mul_concat")
    return torch.ops.acl_ops.sub_mul_concat(x, y)


def pattern_sub_mul_concat(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.cat((x, y, x - y, x * y), dim=2)


patterns = PatternMatcherPass()
inputs = (torch.randn(10, 10, 10), torch.randn(10, 10, 10))
register_replacement(
    pattern_sub_mul_concat,
    fused_sub_mul_concat,
    inputs,
    fwd_only,
    patterns,
)

count = 0


def custom_pass(graph: torch.fx.graph):
    global count
    count = patterns.apply(graph)


inductor_config.post_grad_custom_post_pass = custom_pass


def test_pattern_matcher_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    sub = x - y
    mul = x * y
    cat = torch.cat((x, y, sub, mul), dim=2)
    return cat


import pytest
from typing import Tuple


@pytest.mark.parametrize("shape", [(128, 10, 64), (128, 64, 10)])
def test_pattern_matcher(shape: Tuple[int, int, int]):
    lhs = torch.randn(shape, device="npu")
    rhs = torch.randn(shape, device="npu")

    res = test_pattern_matcher_fn(lhs, rhs)
    compiled_res = torch.compile(
        test_pattern_matcher_fn, backend="inductor", fullgraph=True
    )(lhs, rhs)
    assert torch.allclose(res, compiled_res, rtol=1e-5, atol=1e-5)
    assert count == 1
