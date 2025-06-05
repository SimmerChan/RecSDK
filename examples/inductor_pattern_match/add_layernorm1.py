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

import time
import torch
import torch.nn.functional as F
import torch.fx as fx
import numpy as np
from typing import List, Tuple
import operator

try:
    import torch_npu
except ImportError:
    torch_npu = None
    print("Warning: torch_npu not available, will skip NPU tests")

# 检测NPU环境
npu_env = torch_npu is not None and torch.npu.is_available()
device = "npu:0" if npu_env else "cuda" if torch.cuda.is_available() else "cpu"


def create_fused_add_layernorm(weight, bias, eps):
    """创建特定权重和偏置的融合函数"""

    def fused_op(x1, x2):
        if npu_env:
            return torch_npu.npu_add_layer_norm(x1, x2, weight, bias, eps)[0]
        else:
            added = x1 + x2
            return F.layer_norm(added, weight.shape, weight, bias, eps)

    return fused_op


# 直接的图遍历和替换方法
def apply_add_layernorm_fusion_direct(model):
    """直接遍历图进行Add + LayerNorm融合"""
    traced = fx.symbolic_trace(model)
    graph = traced.graph

    # 查找模式并替换
    nodes_to_remove = []
    match_count = 0

    for node in graph.nodes:
        # 查找LayerNorm调用
        if node.op == "call_module":
            try:
                module = traced.get_submodule(node.target)
                if isinstance(module, torch.nn.LayerNorm):
                    # 检查输入是否来自add操作
                    if len(node.args) == 1 and isinstance(node.args[0], fx.Node):
                        add_node = node.args[0]
                        if (
                            add_node.op == "call_function"
                            and add_node.target == operator.add
                        ):
                            match_count += 1

                            # 获取add操作的输入
                            x1, x2 = add_node.args

                            fused_func = create_fused_add_layernorm(
                                module.weight, module.bias, module.eps
                            )

                            # 在LayerNorm节点之前插入融合操作
                            with graph.inserting_before(node):
                                fused_node = graph.call_function(
                                    fused_func, args=(x1, x2)
                                )

                            # 替换LayerNorm节点的所有使用
                            node.replace_all_uses_with(fused_node)

                            # 标记要删除的节点
                            nodes_to_remove.extend([node, add_node])
            except AttributeError:
                continue

    # 删除被替换的节点
    for node in nodes_to_remove:
        if len(node.users) == 0:  # 确保节点没有被使用
            graph.erase_node(node)

    # 重新编译图
    graph.lint()
    traced.recompile()

    print(f"Total patterns replaced: {match_count}")
    return traced


class AddLayerNorm(torch.nn.Module):
    """Add + LayerNorm模块"""

    def __init__(self):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(768, eps=1e-6)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Add操作
        added = x1 + x2
        # LayerNorm操作
        return self.norm1(added)


class AddLayerNormPerformanceBenchmark:
    """Add + LayerNorm FX变换性能基准测试"""

    def __init__(self, device: str = "cpu"):
        self.device = device

    def create_test_data(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        dtype: torch.dtype = torch.float16,
    ):
        """创建测试数据"""
        x1 = torch.randn(
            batch_size, seq_len, hidden_dim, dtype=dtype, device=self.device
        )
        x2 = torch.randn(
            batch_size, seq_len, hidden_dim, dtype=dtype, device=self.device
        )
        return x1, x2

    def create_original_model(self, hidden_dim: int = 768):
        """创建原始模型"""
        model = AddLayerNorm()
        model.norm1 = torch.nn.LayerNorm(hidden_dim, eps=1e-6)
        return model.to(self.device, dtype=torch.float16).eval()

    def create_fx_transformed_model(self, model):
        """创建FX变换后的模型"""
        return apply_add_layernorm_fusion_direct(model)

    def benchmark_model(
        self, model, inputs: Tuple, warmup_runs: int = 10, benchmark_runs: int = 100
    ) -> Tuple[float, float]:
        """基准测试模型

        Returns:
            Tuple[float, float]: (平均时间(ms), 标准差(ms))
        """
        # 预热
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = model(*inputs)

        if self.device.startswith("npu"):
            torch.npu.synchronize()
        elif self.device.startswith("cuda"):
            torch.cuda.synchronize()

        # 基准测试
        times = []
        for _ in range(benchmark_runs):
            start_time = time.perf_counter()
            with torch.no_grad():
                result = model(*inputs)

            if self.device.startswith("npu"):
                torch.npu.synchronize()
            elif self.device.startswith("cuda"):
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # 转换为毫秒

        return np.mean(times), np.std(times)

    def run_performance_comparison(
        self,
        batch_sizes: List[int] = [1, 4, 8, 16],
        seq_lens: List[int] = [128, 512, 1024],
        hidden_dims: List[int] = [768, 1024, 2048],
    ):
        """运行性能对比测试"""
        print(f"\n=== FX Add + LayerNorm 融合性能对比测试 (Device: {self.device}) ===")
        print(
            f"{'Shape':<20} {'Original (ms)':<15} {'FX Fused (ms)':<15} {'FX Speedup':<12} {'Status':<10}"
        )
        print("-" * 80)

        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                for hidden_dim in hidden_dims:
                    shape_str = f"{batch_size}x{seq_len}x{hidden_dim}"

                    try:
                        # 创建测试数据
                        x1, x2 = self.create_test_data(batch_size, seq_len, hidden_dim)
                        inputs = (x1, x2)

                        # 创建原始模型
                        original_model = self.create_original_model(hidden_dim)

                        # 测试原始模型
                        original_time, original_std = self.benchmark_model(
                            original_model, inputs
                        )
                        original_time_str = f"{original_time:.3f}±{original_std:.3f}"

                        # 测试FX变换模型
                        try:
                            fx_model = self.create_fx_transformed_model(original_model)
                            fx_time, fx_std = self.benchmark_model(fx_model, inputs)
                            fx_speedup = original_time / fx_time
                            fx_time_str = f"{fx_time:.3f}±{fx_std:.3f}"
                            fx_speedup_str = f"{fx_speedup:.2f}x"
                            fx_status = "✓"
                        except Exception as e:
                            fx_time_str = "Error"
                            fx_speedup_str = "N/A"
                            fx_status = "✗"
                            print(f"FX error for {shape_str}: {e}")

                        print(
                            f"{shape_str:<20} {original_time_str:<15} {fx_time_str:<15} "
                            f"{fx_speedup_str:<12} {fx_status:<10}"
                        )

                    except Exception as e:
                        print(f"{shape_str:<20} Error: {e}")

    def test_correctness(
        self, batch_size: int = 2, seq_len: int = 128, hidden_dim: int = 768
    ):
        """测试结果正确性"""
        print(
            f"\n=== FX变换正确性测试 (Shape: {batch_size}x{seq_len}x{hidden_dim}) ==="
        )

        x1, x2 = self.create_test_data(batch_size, seq_len, hidden_dim)

        # 原始模型结果
        original_model = self.create_original_model(hidden_dim)
        with torch.no_grad():
            original_result = original_model(x1, x2)

        # FX变换结果
        try:
            fx_model = self.create_fx_transformed_model(original_model)
            with torch.no_grad():
                fx_result = fx_model(x1, x2)

            # 计算FX与原始的差异
            fx_max_diff = torch.max(torch.abs(original_result - fx_result)).item()
            fx_mean_diff = torch.mean(torch.abs(original_result - fx_result)).item()

            print(f"\n--- FX变换 vs 原始模型 ---")
            print(f"最大绝对误差: {fx_max_diff:.6f}")
            print(f"平均绝对误差: {fx_mean_diff:.6f}")

            tolerance = 1e-3
            if fx_max_diff < tolerance:
                print(f"✓ FX变换正确性测试通过 (tolerance: {tolerance})")
            else:
                print(
                    f"✗ FX变换正确性测试失败 (max_diff: {fx_max_diff} > tolerance: {tolerance})"
                )

        except Exception as e:
            print(f"✗ FX变换测试失败: {e}")

    def run_detailed_analysis(self):
        """运行详细分析"""
        print(f"\n=== 详细性能分析 ===")

        # 测试不同的tensor形状对性能的影响
        shapes = [
            (1, 128, 768),  # 小batch
            (8, 512, 768),  # 中等batch
            (16, 1024, 1024),  # 大batch
            (32, 2048, 2048),  # 超大batch
        ]

        for batch_size, seq_len, hidden_dim in shapes:
            print(f"\n--- Shape: {batch_size}x{seq_len}x{hidden_dim} ---")

            try:
                x1, x2 = self.create_test_data(batch_size, seq_len, hidden_dim)
                inputs = (x1, x2)

                # 原始模型
                original_model = self.create_original_model(hidden_dim)
                original_time, _ = self.benchmark_model(
                    original_model, inputs, warmup_runs=5, benchmark_runs=50
                )

                # FX变换模型
                try:
                    fx_model = self.create_fx_transformed_model(original_model)
                    fx_time, _ = self.benchmark_model(
                        fx_model, inputs, warmup_runs=5, benchmark_runs=50
                    )
                    fx_speedup = original_time / fx_time
                    print(f"原始模型: {original_time:.3f}ms")
                    print(f"FX变换: {fx_time:.3f}ms (加速比: {fx_speedup:.2f}x)")

                    # 如果是NPU环境，还可以测试NPU融合算子
                    if npu_env:
                        print(f"NPU融合算子被调用: {fx_speedup > 1.0}")

                except Exception as e:
                    print(f"FX变换失败: {e}")

            except Exception as e:
                print(f"测试失败: {e}")


def run_performance_tests():
    """运行性能测试"""
    # 检测可用设备
    if torch.cuda.is_available():
        test_device = "cuda"
    elif torch_npu is not None and torch.npu.is_available():
        test_device = "npu:0"
    else:
        test_device = "cpu"

    print(f"使用设备: {test_device}")

    # 创建性能测试实例
    benchmark = AddLayerNormPerformanceBenchmark(test_device)

    # 运行正确性测试
    benchmark.test_correctness()

    # 运行性能对比测试
    benchmark.run_performance_comparison(
        batch_sizes=[1, 4, 8], seq_lens=[128, 512], hidden_dims=[768, 1024]
    )

    # 针对大尺寸的额外测试
    print("\n=== 大尺寸性能测试 ===")
    benchmark.run_performance_comparison(
        batch_sizes=[16, 32], seq_lens=[1024, 2048], hidden_dims=[2048, 4096]
    )

    # 真实场景测试
    print("\n=== 真实场景测试 ===")
    benchmark.run_performance_comparison(
        batch_sizes=[11], seq_lens=[256], hidden_dims=[768]
    )

    # 详细分析
    benchmark.run_detailed_analysis()


def test_fx_transformation():
    """测试FX图变换"""

    model = AddLayerNorm().to(device, dtype=torch.float16).eval()

    # 创建测试数据
    x1 = torch.randn(11, 256, 768, device=device, dtype=torch.float16)
    x2 = torch.randn(11, 256, 768, device=device, dtype=torch.float16)

    # 原始输出
    with torch.no_grad():
        expected = model(x1, x2)

    print("\n=== 原始模型 ===")
    traced_original = fx.symbolic_trace(model)
    print("原始图结构:")
    print(traced_original.graph)
    print("\n原始代码:")
    print(traced_original.code)

    # 方法2: 直接图操作
    print("\n=== 直接图操作 ===")
    direct_model = apply_add_layernorm_fusion_direct(model)

    print("直接操作后图结构:")
    print(direct_model.graph)
    print("\n直接操作后代码:")
    print(direct_model.code)

    with torch.no_grad():
        result2 = direct_model(x1, x2)
    print(
        f"直接操作方法结果匹配: {torch.allclose(result2, expected, rtol=1e-3, atol=1e-3)}"
    )


if __name__ == "__main__":
    # 运行FX变换测试
    test_fx_transformation()

    # 运行性能测试
    print("\n" + "=" * 80)
    print("开始性能测试...")
    print("=" * 80)
    run_performance_tests()
