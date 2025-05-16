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
def process_tensors(input0_arg, input1_arg, activation_fn=tf.nn.relu):
    # Step 1: 对 input0 和 input1 进行矩阵乘法
    # input0 形状是 (128, 1523, 1)，input1 形状是 (32, 1532)
    # 使用 tf.matmul 进行矩阵乘法，得到形状 (32, 128, 1)
    input0_trans = tf.transpose(input0_arg, perm=[1, 0, 2])
    # print(input0_trans.shape, tf.size(input0_trans).numpy())
    input0_reshape = tf.reshape(input0_trans, (1523, 128))
    matmul_output = tf.matmul(input1_arg, input0_reshape)
    matmul_output = tf.reshape(matmul_output, (32, 128, 1))
    # Step 2: 应用激活函数
    # 对 matmul_output 应用激活函数 (默认使用 ReLU)
    activated_output = activation_fn(matmul_output)

    # Step 3: 输出的形状是 (128, 32, 1)
    activated_output = tf.transpose(activated_output, perm=[1, 0, 2])  # 转置使其形状变为 (128, 32, 1)

    return activated_output


# 创建输入张量
input0 = tf.random.normal((128, 1523, 1))  # 形状为 (128, 1523, 1)
input1 = tf.constant(tf.random.normal((32, 1523)))  # 形状为 (32, 1532)

# 调用函数
output_tensor = process_tensors(input0, input1)
