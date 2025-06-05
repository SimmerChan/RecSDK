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
    import torch_npu._inductor
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
        weight = torch.ones(hidden_dim, dtype=dtype, device=self.device)
        bias = torch.zeros(hidden_dim, dtype=dtype, device=self.device)
        eps = 1e-6
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

    def torch_add_layernorm_compiled(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """Inductor 编译的 Add + LayerNorm 实现"""
        if not hasattr(self, "_compiled_fn"):
            # 创建并编译函数
            def _add_layernorm(x1, x2, weight, bias, eps):
                added = x1 + x2
                return F.layer_norm(added, weight.shape, weight, bias, eps)

            self._compiled_fn = torch.compile(_add_layernorm, backend="inductor")

        return self._compiled_fn(x1, x2, weight, bias, eps)

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
            f"{'Shape':<20} {'PyTorch (ms)':<15} {'Inductor (ms)':<15} {'NPU Fused (ms)':<15} {'Inductor Speedup':<15} {'NPU Speedup':<12} {'Status':<10}"
        )
        print("-" * 120)

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

                        # 测试 PyTorch 原生实现
                        torch_time, torch_std = self.benchmark_function(
                            self.torch_add_layernorm, inputs
                        )
                        torch_time_str = f"{torch_time:.3f}±{torch_std:.3f}"

                        # 测试 Inductor 编译实现
                        try:
                            inductor_time, inductor_std = self.benchmark_function(
                                self.torch_add_layernorm_compiled, inputs
                            )
                            inductor_speedup = torch_time / inductor_time
                            inductor_time_str = (
                                f"{inductor_time:.3f}±{inductor_std:.3f}"
                            )
                            inductor_speedup_str = f"{inductor_speedup:.2f}x"
                            inductor_status = "✓"
                        except Exception as e:
                            inductor_time_str = "Error"
                            inductor_speedup_str = "N/A"
                            inductor_status = "✗"
                            print(f"Inductor error for {shape_str}: {e}")

                        # 测试 NPU 融合实现
                        if torch_npu is not None and self.device.startswith("npu"):
                            try:
                                npu_time, npu_std = self.benchmark_function(
                                    self.torch_npu_add_layernorm, inputs
                                )
                                npu_speedup = torch_time / npu_time
                                npu_time_str = f"{npu_time:.3f}±{npu_std:.3f}"
                                npu_speedup_str = f"{npu_speedup:.2f}x"
                                npu_status = "✓"
                            except Exception as e:
                                npu_time_str = "Error"
                                npu_speedup_str = "N/A"
                                npu_status = "✗"
                                print(f"NPU error for {shape_str}: {e}")
                        else:
                            npu_time_str = "N/A"
                            npu_speedup_str = "N/A"
                            npu_status = "Skip"

                        # 综合状态
                        overall_status = (
                            "✓"
                            if inductor_status == "✓" and (npu_status in ["✓", "Skip"])
                            else "✗"
                        )

                        print(
                            f"{shape_str:<20} {torch_time_str:<15} {inductor_time_str:<15} {npu_time_str:<15} "
                            f"{inductor_speedup_str:<15} {npu_speedup_str:<12} {overall_status:<10}"
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

        # PyTorch 原生结果
        torch_result = self.torch_add_layernorm(x1, x2, weight, bias, eps)

        # Inductor 编译结果
        try:
            inductor_result = self.torch_add_layernorm_compiled(
                x1, x2, weight, bias, eps
            )

            # 计算 Inductor 与 PyTorch 的差异
            inductor_max_diff = torch.max(
                torch.abs(torch_result - inductor_result)
            ).item()
            inductor_mean_diff = torch.mean(
                torch.abs(torch_result - inductor_result)
            ).item()

            print(f"\n--- Inductor vs PyTorch ---")
            print(f"最大绝对误差: {inductor_max_diff:.6f}")
            print(f"平均绝对误差: {inductor_mean_diff:.6f}")

            tolerance = 1e-5
            if inductor_max_diff < tolerance:
                print(f"✓ Inductor 正确性测试通过 (tolerance: {tolerance})")
            else:
                print(
                    f"✗ Inductor 正确性测试失败 (max_diff: {inductor_max_diff} > tolerance: {tolerance})"
                )

        except Exception as e:
            print(f"✗ Inductor 测试失败: {e}")

        # NPU 融合结果
        if torch_npu is not None and self.device.startswith("npu"):
            try:
                npu_result = self.torch_npu_add_layernorm(x1, x2, weight, bias, eps)

                # 计算 NPU 与 PyTorch 的差异
                npu_max_diff = torch.max(torch.abs(torch_result - npu_result)).item()
                npu_mean_diff = torch.mean(torch.abs(torch_result - npu_result)).item()

                print(f"\n--- NPU vs PyTorch ---")
                print(f"最大绝对误差: {npu_max_diff:.6f}")
                print(f"平均绝对误差: {npu_mean_diff:.6f}")

                # 相对误差
                npu_rel_error = torch.mean(
                    torch.abs((torch_result - npu_result) / (torch_result + 1e-8))
                ).item()
                print(f"平均相对误差: {npu_rel_error:.6f}")

                # 判断是否通过
                tolerance = 1e-3
                if npu_max_diff < tolerance:
                    print(f"✓ NPU 正确性测试通过 (tolerance: {tolerance})")
                else:
                    print(
                        f"✗ NPU 正确性测试失败 (max_diff: {npu_max_diff} > tolerance: {tolerance})"
                    )

            except Exception as e:
                print(f"✗ NPU 测试失败: {e}")
        else:
            print("\nSkip: NPU 不可用")


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
    
    print("\n=== 真实尺寸测试 ===")
    benchmark.run_comparison(
        batch_sizes=[11], seq_lens=[256], hidden_dims=[768]
    )


if __name__ == "__main__":
    main()
