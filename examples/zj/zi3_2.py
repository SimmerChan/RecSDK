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
import numpy as np
import tensorflow as tf

# 禁用 TensorFlow 2.x 的急切执行模式
tf.compat.v1.disable_eager_execution()


def build_graph(input1, input2, input3, input4):
    # 1. matmul [128,128] [128,2560] = [128,2560]
    #    matmul [128,1861] [1861,2560] = [128,2560]
    matmul1 = tf.matmul(input1, input2)
    matmul2 = tf.matmul(input3, input4)
    # 2. bias_add [128,2560] [2560] = [128,2560]
    constant1 = tf.constant(3.0, shape=[2560])
    bias_add1 = tf.nn.bias_add(matmul1, constant1)
    constant2 = tf.constant(10.0, shape=[2560])
    bias_add2 = tf.nn.bias_add(matmul2, constant2)
    # 3. sigmoid [128,2560]
    sigmoid_output = tf.nn.sigmoid(bias_add1)
    # 4. mul [128,2560] [128,2560] = [128,2560]
    mul1 = tf.multiply(sigmoid_output, bias_add2)
    # 5. mul [128,2560] [128,2560] = [128,2560]
    mul2 = mul1 * 5.0
    # 6. reduce_mean [128,2560] = [128]
    reduce_mean1 = tf.math.reduce_mean(mul2, axis=1, keepdims=True)
    # 7. stop_gradiant [128]
    stop_gradiant_output = tf.stop_gradient(reduce_mean1)
    # 8. squared_difference [128] [128,2560] = [128, 2560]
    squared_difference_output = tf.math.squared_difference(mul2, stop_gradiant_output)
    # 9. reduce_mean [128,2560] = [128]
    reduce_mean2 = tf.reduce_mean(squared_difference_output, axis=1)
    # 10. add [128]
    add_result = reduce_mean2 + 5.0
    # 11. resqrt [128]
    rsqrt_output = tf.compat.v1.math.rsqrt(add_result)
    # 11. mul [128]
    mul3 = rsqrt_output * 11.11
    # 12. mul [128] [128] = [128]
    return reduce_mean1 * mul3


if __name__ == '__main__':
    # 创建输入张量
    input1_shape = [128, 128]
    input2_shape = [128, 2560]
    input3_shape = [128, 1861]
    input4_shape = [1861, 2560]
    input1 = tf.compat.v1.placeholder(tf.float32, shape=input1_shape)
    input2 = tf.compat.v1.placeholder(tf.float32, shape=input2_shape)
    input3 = tf.compat.v1.placeholder(tf.float32, shape=input3_shape)
    input4 = tf.compat.v1.placeholder(tf.float32, shape=input4_shape)

    # 使用 XLA 编译
    # 注意：在 TensorFlow 1.x 中，XLA 编译需要通过 tf.xla.experimental.compile 实现
    # 或者通过设置环境变量启用

    # 创建会话并运行计算
    with tf.compat.v1.Session() as sess:
        i1 = np.random.triangular(left=100, mode=500, right=1000, size=input1_shape)
        i2 = np.random.triangular(left=100, mode=500, right=1000, size=input2_shape)
        i3 = np.random.triangular(left=100, mode=500, right=1000, size=input3_shape)
        i4 = np.random.triangular(left=100, mode=500, right=1000, size=input4_shape)

        matmul1 = np.matmul(i1, i2)
        matmul2 = np.matmul(i3, i4)
        constant1 = np.full(2560, 3.0)
        bias_add1 = matmul1 + constant1
        constant2 = np.full(2560, 10.0)
        bias_add2 = matmul2 + constant2
        sigmoid_output = 1 / (1 + np.exp(-bias_add1))
        mul1 = sigmoid_output * bias_add2
        mul2 = mul1 * 5.0
        reduce_mean1 = np.mean(mul2, axis=1, keepdims=True)
        stop_gradiant_output = reduce_mean1.copy()
        squared_difference_output = (mul2 - stop_gradiant_output) ** 2
        reduce_mean2 = np.mean(squared_difference_output, axis=1)
        add_result = reduce_mean2 + 5.0
        rsqrt_output = 1 / np.sqrt(add_result + 1e-8)
        mul3 = rsqrt_output * 11.11
        cpu_output = reduce_mean1 * mul3

        # 运行 XLA 编译后的计算图
        output_graph = build_graph(input1, input2, input3, input4)
        output_xla = sess.run(
            output_graph, feed_dict={input1: i1, input2: i2, input3: i3, input4: i4}
        )
        output_xla_real = np.array(output_xla).reshape((-1,))
        output_expect = cpu_output.reshape((-1,))
