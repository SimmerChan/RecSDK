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
import numpy as np
from typing import List, Tuple

try:
    import torch_npu
except ImportError:
    torch_npu = None
    print("Warning: torch_npu not available, will skip NPU tests")


class AddLayerNormBenchmark:
    """Add + LayerNorm 性能基准测试"""

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
        weight = torch.randn(hidden_dim, dtype=dtype, device=self.device)
        bias = torch.randn(hidden_dim, dtype=dtype, device=self.device)
        eps = 1e-5
        return x1, x2, weight, bias, eps

    def torch_add_layernorm(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """标准 PyTorch Add + LayerNorm 实现"""
        added = x1 + x2
        return F.layer_norm(added, weight.shape, weight, bias, eps)

    def torch_npu_add_layernorm(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """NPU 融合 Add + LayerNorm 实现"""
        if torch_npu is None:
            raise RuntimeError("torch_npu is not available")
        return torch_npu.npu_add_layer_norm(x1, x2, weight, bias, eps)[0]

    def benchmark_function(
        self, func, inputs: Tuple, warmup_runs: int = 10, benchmark_runs: int = 100
    ) -> Tuple[float, float]:
        """基准测试函数

        Returns:
            Tuple[float, float]: (平均时间(ms), 标准差(ms))
        """
        # 预热
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = func(*inputs)

        if self.device.startswith("npu"):
            torch.npu.synchronize()
        elif self.device.startswith("cuda"):
            torch.cuda.synchronize()

        # 基准测试
        times = []
        for _ in range(benchmark_runs):
            start_time = time.perf_counter()
            with torch.no_grad():
                result = func(*inputs)

            if self.device.startswith("npu"):
                torch.npu.synchronize()
            elif self.device.startswith("cuda"):
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # 转换为毫秒

        return np.mean(times), np.std(times)

    def run_comparison(
        self,
        batch_sizes: List[int] = [1, 4, 8, 16],
        seq_lens: List[int] = [128, 512, 1024],
        hidden_dims: List[int] = [768, 1024, 2048],
    ):
        """运行性能对比测试"""
        print(f"\n=== Add + LayerNorm 性能对比测试 (Device: {self.device}) ===")
        print(
            f"{'Shape':<20} {'PyTorch (ms)':<15} {'NPU Fused (ms)':<15} {'Speedup':<10} {'Status':<10}"
        )
        print("-" * 80)

        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                for hidden_dim in hidden_dims:
                    shape_str = f"{batch_size}x{seq_len}x{hidden_dim}"

                    try:
                        # 创建测试数据
                        x1, x2, weight, bias, eps = self.create_test_data(
                            batch_size, seq_len, hidden_dim
                        )
                        inputs = (x1, x2, weight, bias, eps)

                        # 测试 PyTorch 实现
                        torch_time, torch_std = self.benchmark_function(
                            self.torch_add_layernorm, inputs
                        )

                        # 测试 NPU 融合实现
                        if torch_npu is not None and self.device.startswith("npu"):
                            try:
                                npu_time, npu_std = self.benchmark_function(
                                    self.torch_npu_add_layernorm, inputs
                                )
                                speedup = torch_time / npu_time
                                status = "✓"
                                npu_time_str = f"{npu_time:.3f}±{npu_std:.3f}"
                                speedup_str = f"{speedup:.2f}x"
                            except Exception as e:
                                npu_time_str = "Error"
                                speedup_str = "N/A"
                                status = "✗"
                                print(f"NPU error for {shape_str}: {e}")
                        else:
                            npu_time_str = "N/A"
                            speedup_str = "N/A"
                            status = "Skip"

                        torch_time_str = f"{torch_time:.3f}±{torch_std:.3f}"
                        print(
                            f"{shape_str:<20} {torch_time_str:<15} {npu_time_str:<15} {speedup_str:<10} {status:<10}"
                        )

                    except Exception as e:
                        print(f"{shape_str:<20} Error: {e}")

    def test_correctness(
        self, batch_size: int = 2, seq_len: int = 128, hidden_dim: int = 768
    ):
        """测试结果正确性"""
        print(f"\n=== 正确性测试 (Shape: {batch_size}x{seq_len}x{hidden_dim}) ===")

        x1, x2, weight, bias, eps = self.create_test_data(
            batch_size, seq_len, hidden_dim
        )

        # PyTorch 结果
        torch_result = self.torch_add_layernorm(x1, x2, weight, bias, eps)

        if torch_npu is not None and self.device.startswith("npu"):
            try:
                # NPU 结果
                npu_result = self.torch_npu_add_layernorm(x1, x2, weight, bias, eps)

                # 计算差异
                max_diff = torch.max(torch.abs(torch_result - npu_result)).item()
                mean_diff = torch.mean(torch.abs(torch_result - npu_result)).item()

                print(f"最大绝对误差: {max_diff:.6f}")
                print(f"平均绝对误差: {mean_diff:.6f}")

                # 相对误差
                rel_error = torch.mean(
                    torch.abs((torch_result - npu_result) / (torch_result + 1e-8))
                ).item()
                print(f"平均相对误差: {rel_error:.6f}")

                # 判断是否通过
                tolerance = 1e-3
                if max_diff < tolerance:
                    print(f"✓ 正确性测试通过 (tolerance: {tolerance})")
                else:
                    print(
                        f"✗ 正确性测试失败 (max_diff: {max_diff} > tolerance: {tolerance})"
                    )

            except Exception as e:
                print(f"✗ NPU 测试失败: {e}")
        else:
            print("Skip: NPU 不可用")


def main():
    """主函数"""
    # 检测可用设备
    if torch.cuda.is_available():
        device = "cuda"
    elif torch_npu is not None and torch.npu.is_available():
        device = "npu:0"
    else:
        device = "cpu"

    print(f"使用设备: {device}")

    # 创建基准测试实例
    benchmark = AddLayerNormBenchmark(device)

    # 运行正确性测试
    benchmark.test_correctness()

    # 运行性能对比测试
    benchmark.run_comparison(
        batch_sizes=[1, 4, 8], seq_lens=[128, 512], hidden_dims=[768, 1024]
    )

    # 针对大尺寸的额外测试
    print("\n=== 大尺寸性能测试 ===")
    benchmark.run_comparison(
        batch_sizes=[16, 32], seq_lens=[1024, 2048], hidden_dims=[2048, 4096]
    )


if __name__ == "__main__":
    main()
