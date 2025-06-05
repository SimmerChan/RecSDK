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
import torch.fx as fx
import operator

try:
    import torch_npu

    npu_env = True
except ImportError:
    npu_env = False

device = "cuda" if torch.cuda.is_available() else "cpu"
if npu_env:
    device = "npu"
print(f"device: {device}")


def create_fused_add_layernorm(weight, bias, eps):
    """创建特定权重和偏置的融合函数"""

    def fused_op(x1, x2):
        print("fused_add_layernorm called!")
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
                            print(f"Replacing Add + LayerNorm pattern #{match_count}")

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
                            nodes_to_remove.extend([add_node, node])
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
    # 运行测试
    test_fx_transformation()
