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

from typing import Tuple, Optional
import torch
import torch.nn.functional as F
from torch._inductor.pattern_matcher import (
    fwd_only,
    PatternMatcherPass,
    register_replacement,
)
import torch._inductor.config as inductor_config
import torch_npu
import torch_npu._inductor


def fused_add_layer_norm(
    x1: torch.Tensor,
    x2: torch.Tensor,
    normalized_shape: Tuple[int, ...],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """使用torch_npu.npu_add_layer_norm融合算子"""
    return torch_npu.npu_add_layer_norm(x1, x2, normalized_shape, weight, bias, eps)[0]


def pattern_add_layer_norm(
    x1: torch.Tensor,
    x2: torch.Tensor,
    normalized_shape: Tuple[int, ...],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """原始的Add + LayerNorm模式"""
    # Add操作
    added = x1 + x2
    # LayerNorm操作
    return F.layer_norm(added, normalized_shape, weight, bias, eps)


def pattern_add_cast_layer_norm(
    x1: torch.Tensor,
    x2: torch.Tensor,
    normalized_shape: Tuple[int, ...],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Add + Cast + LayerNorm模式"""
    # Add操作
    added = x1 + x2
    # Cast操作（可选的精度提升）
    casted = added.to(dtype)
    # LayerNorm操作
    return F.layer_norm(casted, normalized_shape, weight, bias, eps)


def check_add_layernorm_constraints(match):
    """检查Add + LayerNorm融合的约束条件"""
    # 获取输入张量
    x1_node = match.kwargs.get("x1") or (match.args[0] if len(match.args) > 0 else None)
    x2_node = match.kwargs.get("x2") or (match.args[1] if len(match.args) > 1 else None)

    if x1_node is None or x2_node is None:
        return False

    # 检查输入张量的数据类型约束
    # 根据文档：不支持输入x1和x2的数据类型均为fp32
    if hasattr(x1_node, "meta") and hasattr(x2_node, "meta"):
        x1_dtype = getattr(x1_node.meta.get("tensor_meta", {}), "dtype", None)
        x2_dtype = getattr(x2_node.meta.get("tensor_meta", {}), "dtype", None)

        # 如果两个输入都是fp32，则不融合
        if x1_dtype == torch.float32 and x2_dtype == torch.float32:
            return False

    # 检查gamma和beta的shape约束
    weight_node = match.kwargs.get("weight")
    bias_node = match.kwargs.get("bias")

    if weight_node is not None and hasattr(weight_node, "meta"):
        weight_shape = getattr(weight_node.meta.get("tensor_meta", {}), "shape", None)
        if weight_shape is not None and len(weight_shape) != 1:
            return False

    if bias_node is not None and hasattr(bias_node, "meta"):
        bias_shape = getattr(bias_node.meta.get("tensor_meta", {}), "shape", None)
        if bias_shape is not None and len(bias_shape) != 1:
            return False

    return True


# 创建模式匹配器
patterns = PatternMatcherPass()

# 示例输入用于模式匹配
batch_size, seq_len, hidden_dim = 2, 128, 768
inputs_basic = (
    torch.randn(batch_size, seq_len, hidden_dim),
    torch.randn(batch_size, seq_len, hidden_dim),
    (hidden_dim,),  # normalized_shape
    torch.randn(hidden_dim),  # weight
    torch.randn(hidden_dim),  # bias
    1e-5,  # eps
)

# 注册基本的Add + LayerNorm模式
register_replacement(
    pattern_add_layer_norm,
    fused_add_layer_norm,
    inputs_basic,
    fwd_only,
    patterns,
    extra_check=check_add_layernorm_constraints,
)

# 注册Add + Cast + LayerNorm模式的输入
inputs_with_cast = (
    torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16),
    torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16),
    (hidden_dim,),  # normalized_shape
    torch.randn(hidden_dim),  # weight
    torch.randn(hidden_dim),  # bias
    1e-5,  # eps
    torch.float32,  # dtype for cast
)

register_replacement(
    pattern_add_cast_layer_norm,
    lambda x1,
    x2,
    normalized_shape,
    weight=None,
    bias=None,
    eps=1e-5,
    dtype=torch.float32: fused_add_layer_norm(
        x1, x2, normalized_shape, weight, bias, eps
    ),
    inputs_with_cast,
    fwd_only,
    patterns,
    extra_check=check_add_layernorm_constraints,
)

count = 0


def custom_add_layernorm_pass(graph: torch.fx.graph):
    """自定义的Add + LayerNorm融合pass"""
    global count
    count = patterns.apply(graph)
    return count


# 设置在pre_grad阶段执行，确保在inductor生成triton算子之前进行模式匹配
inductor_config.pre_grad_custom_pass = custom_add_layernorm_pass


def test_add_layernorm_pattern():
    """测试Add + LayerNorm模式匹配"""

    def model_with_add_layernorm(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Add操作
        added = x1 + x2
        # LayerNorm操作
        return F.layer_norm(added, (768,), weight=None, bias=None, eps=1e-5)

    # 创建测试数据
    x1 = torch.randn(2, 128, 768, device="npu", dtype=torch.float16)
    x2 = torch.randn(2, 128, 768, device="npu", dtype=torch.float16)

    # 原始输出
    expected = model_with_add_layernorm(x1, x2)

    # 编译后的输出
    compiled_model = torch.compile(
        model_with_add_layernorm, backend="inductor", fullgraph=True
    )
    actual = compiled_model(x1, x2)

    # 验证结果一致性
    assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-4)
    print(f"Add + LayerNorm pattern matched {count} times")

    return actual


if __name__ == "__main__":
    # 运行测试
    test_add_layernorm_pattern()
