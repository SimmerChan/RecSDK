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

import os
import torch
import logging

# 配置logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

device = "npu" if torch.npu.is_available() else "cpu"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

import torch_npu._inductor


def simple_model(x, y):
    """简单的模型函数，包含基本的张量操作"""
    z = x + y
    w = torch.relu(z)
    return w * 2


def complex_model(x, y):
    """稍微复杂的模型，包含多种操作"""
    # 矩阵乘法
    z1 = torch.matmul(x, y.transpose(-2, -1))
    # 激活函数
    z2 = torch.relu(z1)
    # 归一化
    z3 = torch.softmax(z2, dim=-1)
    # 求和
    z4 = torch.sum(z3, dim=-1, keepdim=True)
    return z4


def test_inductor_simple_compilation():
    """测试torch inductor的基本编译功能"""
    assert device == "npu"
    # 创建测试数据
    x = torch.randn(4, 8, device=device)
    y = torch.randn(4, 8, device=device)

    # 原始模型输出
    expected = simple_model(x, y)

    # 使用inductor编译模型
    compiled_model = torch.compile(simple_model, backend="inductor")

    # 编译后模型输出
    actual = compiled_model(x, y)

    # 验证结果一致性
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_inductor_complex_compilation():
    """测试torch inductor编译复杂模型"""
    assert device == "npu"
    # 创建测试数据
    batch_size, seq_len, hidden_dim = 2, 16, 32
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    y = torch.randn(batch_size, hidden_dim, seq_len, device=device)

    # 原始模型输出
    expected = complex_model(x, y)

    # 使用inductor编译模型
    compiled_model = torch.compile(complex_model, backend="inductor")

    # 编译后模型输出
    actual = compiled_model(x, y)

    # 验证结果一致性
    assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-4)


def test_inductor_performance_comparison():
    """测试torch inductor的性能对比"""
    assert device == "npu"
    import time

    # 创建较大的测试数据
    x = torch.randn(100, 100, device=device)
    y = torch.randn(100, 100, device=device)

    # 预热
    for _ in range(5):
        simple_model(x, y)

    # 测试原始模型性能
    start_time = time.time()
    for _ in range(100):
        result_original = simple_model(x, y)
    original_time = time.time() - start_time

    # 编译模型
    compiled_model = torch.compile(simple_model, backend="inductor")

    # 预热编译后的模型
    for _ in range(5):
        compiled_model(x, y)

    # 测试编译后模型性能
    start_time = time.time()
    for _ in range(100):
        result_compiled = compiled_model(x, y)
    compiled_time = time.time() - start_time

    # 验证结果一致性
    assert torch.allclose(result_original, result_compiled, rtol=1e-5, atol=1e-5)

    logger.info(f"Original model time: {original_time:.4f}s")
    logger.info(f"Compiled model time: {compiled_time:.4f}s")
    logger.info(f"Speedup: {original_time / compiled_time:.2f}x")


if __name__ == "__main__":
    # 运行所有测试
    test_inductor_simple_compilation()
    test_inductor_complex_compilation()
    test_inductor_performance_comparison()
