#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.
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
# ==============================================================================
import tensorflow as tf

#tf.config.optimizer.set_jit(True) # 启用 XLA JIT 编译加速
@tf.function(jit_compile=True)
def process_tensor(input_tensor):
    # Step 1: 使用 reduce_sum 对第二个维度进行求和，结果形状为 (128, 1, 16)
    reduced_tensor = tf.reduce_sum(input_tensor, axis=1, keepdims=True)

    # Step 2: 使用 concat 扩展到 (128, 120, 16)
    # 我们需要将张量沿着第二个维度（axis=1）进行扩展
    #expanded_tensor = tf.concat([reduced_tensor] * 120, axis=1)
    expanded_tensor = tf.tile(reduced_tensor, [1, 120, 1])

    return expanded_tensor

# 创建一个形状为 (128, 5, 16) 的输入张量
input_tensor = tf.random.normal((128, 5, 16))

# 调用处理函数
output_tensor = process_tensor(input_tensor)

# 打印输出张量的形状，确保它是 (128, 120, 16)
print("Output shape:", output_tensor.shape)  # 应该输出 (128, 120, 16)
