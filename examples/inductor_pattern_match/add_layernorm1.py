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


# 定义融合操作函数
def fused_add_layernorm(x1, x2, weight, bias, eps=1e-6):
    """融合的Add + LayerNorm实现"""
    print("fused_add_layernorm called!")
    if npu_env:
        return torch_npu.npu_add_layer_norm(x1, x2, weight, bias, eps)[0]
    else:
        # CPU/CUDA fallback
        added = x1 + x2
        return F.layer_norm(added, weight.shape, weight, bias, eps)


# 自定义FX图变换器
class AddLayerNormTransformer(fx.Transformer):
    """自定义的Add + LayerNorm融合变换器"""

    def __init__(self, module):
        super().__init__(module)
        self.match_count = 0
        self.replaced_nodes = set()

    def call_module(self, target, args, kwargs):
        # 检查是否是LayerNorm模块
        try:
            module = self.module.get_submodule(target)
        except AttributeError:
            return super().call_module(target, args, kwargs)

        if isinstance(module, torch.nn.LayerNorm):
            # 检查输入是否来自add操作
            if len(args) == 1 and isinstance(args[0], fx.Node):
                add_node = args[0]
                if (
                    add_node.op == "call_function"
                    and add_node.target == operator.add
                    and add_node not in self.replaced_nodes
                ):
                    # 找到了Add + LayerNorm模式
                    self.match_count += 1
                    print(f"Found Add + LayerNorm pattern #{self.match_count}")

                    # 获取add操作的输入
                    x1, x2 = add_node.args

                    # 获取LayerNorm参数
                    weight = module.weight
                    bias = module.bias
                    eps = module.eps

                    # 标记节点已被替换
                    self.replaced_nodes.add(add_node)
                    self.replaced_nodes.add(self.node)

                    # 创建融合操作调用
                    return self.call_function(
                        fused_add_layernorm, (x1, x2, weight, bias, eps), {}
                    )

        # 默认处理
        return super().call_module(target, args, kwargs)


# 使用subgraph_rewriter的方式
def apply_add_layernorm_fusion_with_rewriter(model):
    """使用subgraph_rewriter应用Add + LayerNorm融合"""
    from torch.fx.subgraph_rewriter import replace_pattern

    # 定义模式
    def pattern(x1, x2, norm_weight, norm_bias, eps):
        add_result = x1 + x2
        return F.layer_norm(add_result, norm_weight.shape, norm_weight, norm_bias, eps)

    # 定义替换
    def replacement(x1, x2, norm_weight, norm_bias, eps):
        return fused_add_layernorm(x1, x2, norm_weight, norm_bias, eps)

    # 首先将LayerNorm模块转换为functional形式
    traced = fx.symbolic_trace(model)

    # 手动转换LayerNorm模块调用为functional调用
    class LayerNormToFunctional(fx.Transformer):
        def call_module(self, target, args, kwargs):
            module = self.module.get_submodule(target)
            if isinstance(module, torch.nn.LayerNorm):
                input_tensor = args[0]
                return F.layer_norm(
                    input_tensor,
                    module.weight.shape,
                    module.weight,
                    module.bias,
                    module.eps,
                )
            return super().call_module(target, args, kwargs)

    # 转换为functional形式
    functional_model = LayerNormToFunctional(traced).transform()

    # 应用模式替换
    replaced_model = replace_pattern(functional_model, pattern, replacement)

    return replaced_model


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

                            # 获取LayerNorm参数
                            weight = module.weight
                            bias = module.bias
                            eps = module.eps

                            # 在add节点之后插入融合操作
                            with graph.inserting_after(add_node):
                                fused_node = graph.call_function(
                                    fused_add_layernorm,
                                    args=(x1, x2, weight, bias, eps),
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

    # 方法1: 使用Transformer
    print("\n=== 方法1: 使用Transformer ===")
    transformer = AddLayerNormTransformer(traced_original)
    transformed_model = transformer.transform()

    print("变换后图结构:")
    print(transformed_model.graph)
    print("\n变换后代码:")
    print(transformed_model.code)

    with torch.no_grad():
        result1 = transformed_model(x1, x2)
    print(
        f"Transformer方法结果匹配: {torch.allclose(result1, expected, rtol=1e-3, atol=1e-3)}"
    )
    print(f"匹配到的模式数量: {transformer.match_count}")

    # 方法2: 直接图操作
    print("\n=== 方法2: 直接图操作 ===")
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

    return result1, result2


if __name__ == "__main__":
    # 运行测试
    test_fx_transformation()
