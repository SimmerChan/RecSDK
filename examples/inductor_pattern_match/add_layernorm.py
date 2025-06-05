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

from typing import Tuple, Optional, Sequence
import torch
import torch.nn.functional as F
from torch._inductor.pattern_matcher import (
    fwd_only,
    PatternMatcherPass,
    register_replacement,
)
import torch._inductor.config as inductor_config
# import torch_npu
# import torch_npu._inductor


# 在文件顶部添加导入
from operator import add
from torch.nn.functional import layer_norm
import torch.fx

# 包装函数以确保正确追踪
torch.fx.wrap('add')
torch.fx.wrap('layer_norm')


def pattern_add_layer_norm(
    x1: torch.Tensor,
    x2: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """原始的Add + LayerNorm模式 - 精确匹配图结构"""
    # 使用operator.add而不是+操作符
    added = add(x1, x2)
    # 直接调用layer_norm函数，使用位置参数
    return layer_norm(added, normalized_shape, weight, bias, eps)


def fused_add_layer_norm(
    x1: torch.Tensor,
    x2: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """融合的Add + LayerNorm实现"""
    print("fused_add_layer_norm called!")
    # return torch_npu.npu_add_layer_norm(x1, x2, weight, bias, eps)[0]

    # 临时实现
    added = add(x1, x2)
    return layer_norm(added, normalized_shape, weight, bias, eps)


# 创建模式匹配器
patterns = PatternMatcherPass()

# 示例输入用于模式匹配 - 确保eps值一致
batch_size, seq_len, hidden_dim = 2, 128, 768
inputs_basic = (
    torch.randn(batch_size, seq_len, hidden_dim),  # x1
    torch.randn(batch_size, seq_len, hidden_dim),  # x2
    (768,),  # normalized_shape
    torch.randn(hidden_dim),  # weight
    torch.randn(hidden_dim),  # bias
    1e-5,  # eps - 修改为与测试用例一致
)

# 注册基本的Add + LayerNorm模式
register_replacement(
    pattern_add_layer_norm,
    fused_add_layer_norm,
    inputs_basic,
    fwd_only,
    patterns,
)

count = 0


def custom_add_layernorm_pass(graph: torch.fx.graph):
    """自定义的Add + LayerNorm融合pass"""
    global count

    print("\n=== Before Pattern Matching ===")
    print(f"Graph nodes: {len(list(graph.nodes))}")
    for i, node in enumerate(graph.nodes):
        print(
            f"Node {i}: {node.op} - {node.target} - args: {node.args} - kwargs: {node.kwargs}"
        )

    count = patterns.apply(graph)

    print(f"\n=== After Pattern Matching ===")
    print(f"Patterns matched: {count}")
    print(f"Graph nodes: {len(list(graph.nodes))}")

    return count


# 设置在post_grad阶段执行
# 修改注册时机
# inductor_config.post_grad_custom_post_pass = custom_add_layernorm_pass  # 删除这行
inductor_config.pre_grad_custom_pass = custom_add_layernorm_pass  # 添加这行


def test_add_layernorm_pattern():
    """测试Add + LayerNorm模式匹配"""

    def model_with_add_layernorm(
        x1: torch.Tensor,
        x2: torch.Tensor,
        normalized_shape: Sequence[int],
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        # Add操作
        added = x1 + x2
        # LayerNorm操作 - 使用位置参数确保与pattern匹配
        return F.layer_norm(added, normalized_shape, weight, bias, eps)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 创建测试数据
    x1 = torch.randn(2, 128, 768, device=device, dtype=torch.float16)
    x2 = torch.randn(2, 128, 768, device=device, dtype=torch.float16)
    weight = torch.randn(768, device=device, dtype=torch.float16)
    bias = torch.randn(768, device=device, dtype=torch.float16)

    # 原始输出
    expected = model_with_add_layernorm(x1, x2, (768,), weight, bias, 1e-5)

    import torch.fx as fx

    # 符号化追踪模型
    traced = fx.symbolic_trace(model_with_add_layernorm)

    # 打印图结构
    print(traced.graph)

    # 打印生成的代码
    print(traced.code)

    # 编译后的输出
    compiled_model = torch.compile(
        model_with_add_layernorm, backend="inductor", fullgraph=True
    )
    actual = compiled_model(x1, x2, (768,), weight, bias, 1e-5)

    # 验证结果一致性
    print(torch.allclose(actual, expected, rtol=1e-4, atol=1e-4))
    print(f"Add + LayerNorm pattern matched {count} times")

    return actual


if __name__ == "__main__":
    # 运行测试
    test_add_layernorm_pattern()
