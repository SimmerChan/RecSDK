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


def check_tensor_constraints(match) -> bool:
    """检查输入张量是否满足约束条件：维度相同且必须为三维"""
    # 获取匹配的节点参数
    args = match.args
    if len(args) != 2:
        return False

    x_node, y_node = args[0], args[1]

    # 检查是否有形状信息
    if not hasattr(x_node, "meta") or not hasattr(y_node, "meta"):
        return False

    x_shape = x_node.meta.get("tensor_meta")
    y_shape = y_node.meta.get("tensor_meta")

    if x_shape is None or y_shape is None:
        return False

    # 检查是否为三维张量
    if len(x_shape.shape) != 3 or len(y_shape.shape) != 3:
        return False

    # 检查形状是否相同
    if x_shape.shape != y_shape.shape:
        return False

    return True


patterns = PatternMatcherPass()
inputs = (torch.randn(10, 10, 10), torch.randn(10, 10, 10))
register_replacement(
    pattern_sub_mul_concat,
    fused_sub_mul_concat,
    inputs,
    fwd_only,
    patterns,
    # extra_check=check_tensor_constraints,
)

count = 0


def custom_pass(graph: torch.fx.graph):
    global count
    count = patterns.apply(graph)


inductor_config.post_grad_custom_post_pass = custom_pass


def test_pattern_matcher(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    sub = x - y
    mul = x * y
    cat = torch.cat((x, y, sub, mul), dim=2)
    return cat


if __name__ == "__main__":
    x = torch.randn((128, 10, 32), device="npu")
    y = torch.randn((128, 10, 32), device="npu")

    res = test_pattern_matcher(x, y)
    compiled_res = torch.compile(
        test_pattern_matcher, backend="inductor", fullgraph=True
    )(x, y)
    assert torch.allclose(res, compiled_res, rtol=1e-5, atol=1e-5)
    assert count == 1
