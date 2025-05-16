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


@tf.function(jit_compile=True)  # 启用 XLA JIT 编译
def process_tensors(input_first, input_second):
    # Step 1: 执行逐元素减法（sub）
    sub_output = tf.subtract(input_first, input_second, name="sub_output")

    # Step 2: 执行逐元素乘法（mul）
    mul_output = tf.multiply(input_first, input_second, name="mul_output")

    # Step 3: 在 axis=2 上进行 concat
    output = tf.concat([input_first, input_second, sub_output, mul_output], axis=2)

    return output


# 创建两个形状为 (128, 128, 144) 的输入张量
input0 = tf.random.normal((128, 128, 144))
input1 = tf.random.normal((128, 128, 144))

# 调用处理函数
output_tensor = process_tensors(input0, input1)

# 打印输出张量的形状，确保它是 (128, 128, 576)
print("Output shape:", output_tensor.shape)  # 应该输出 (128, 128, 576)
