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

tf.compat.v1.disable_eager_execution()


def numpy_bias_add(x, bias, data_format="NHWC"):
    if data_format == "NHWC":
        # 默认最后一个维度是 channels
        bias_shape = (1,) * (x.ndim - 1) + (-1,)
    elif data_format == "NCHW":
        # 第二个维度是 channels
        bias_shape = (1, -1) + (1,) * (x.ndim - 2)
    else:
        raise ValueError("Unsupported data format")
    return x + bias.reshape(bias_shape)


def build_graph(input1, input2, input3, input4):
    # 1. matmul [128,512] [512,8192] = [128, 8192]
    matmul_output = tf.matmul(input1, input2)
    # 2. bias_add [128,8192] [8192] = [128, 8192]
    bias_add_output = tf.nn.bias_add(matmul_output, input3)
    # 3. mul [128,8192] [128,8192] = [128, 8192]
    mul_output = tf.multiply(bias_add_output, input4)
    # 4. sigmoid [128,8192]
    sigmoid_output = tf.nn.sigmoid(mul_output)
    # 5. less_equal [128,8192]
    constant1 = tf.constant(0.75, shape=(128, 8192))
    less_equal_output = tf.less_equal(sigmoid_output, constant1)
    # 6. zeros_like [128,8192]
    zeros_like_output = tf.zeros_like(sigmoid_output)
    # 7. select [128,8192]
    select_output = tf.raw_ops.Select(condition=less_equal_output, x=zeros_like_output, y=sigmoid_output)
    # 8. sub [128,8192]
    sub_output = tf.subtract(select_output, sigmoid_output)
    # 9. stop_gradiant [128,8192]
    stop_gradiant_output = tf.stop_gradient(sub_output)
    # 10. add [128,8192]
    add_output = sigmoid_output + stop_gradiant_output
    # 11. reshape [128,8192] = [128,64,128]
    reshape_output = tf.reshape(add_output, [128, 64, 128])
    return select_output


if __name__ == '__main__':
    # 创建输入张量
    input1_shape = [128, 512]
    input2_shape = [512, 8192]
    input3_shape = [8192]
    input4_shape = [128, 8192]
    input1 = tf.compat.v1.placeholder(tf.float32, shape=input1_shape)
    input2 = tf.compat.v1.placeholder(tf.float32, shape=input2_shape)
    input3 = tf.compat.v1.placeholder(tf.float32, shape=input3_shape)
    input4 = tf.compat.v1.placeholder(tf.float32, shape=input4_shape)

    # 创建会话并运行计算
    with tf.compat.v1.Session() as sess:
        i1 = np.random.triangular(left=100, mode=500, right=1000, size=input1_shape)
        i2 = np.random.triangular(left=100, mode=500, right=1000, size=input2_shape)
        i3 = np.random.triangular(left=100, mode=500, right=1000, size=input3_shape)
        i4 = np.random.triangular(left=100, mode=500, right=1000, size=input4_shape)

        matmul_output = np.matmul(i1, i2)
        bias_output = numpy_bias_add(matmul_output, i3)
        mul_output = np.multiply(bias_output, i4)
        sig_output = 1 / (1 + np.exp(-mul_output))
        constant1 = 0.75 * np.ones((128, 8192), dtype=np.float32)
        less_output = np.less_equal(sig_output, constant1)
        zeros_output = np.zeros_like(sig_output)
        select_output = np.where(less_output, zeros_output, sig_output)
        sub_output = np.subtract(select_output, sig_output)
        stop_output = sub_output
        add_output = np.sum([sig_output, stop_output], axis=0)
        cpu_output = np.reshape(add_output, [128, 64, 128])

        # 运行 XLA 编译后的计算图
        output_graph = build_graph(input1, input2, input3, input4)
        output_xla = sess.run(
            output_graph, feed_dict={input1: i1, input2: i2, input3: i3, input4: i4}
        )
        output_xla_real = np.array(output_xla).reshape((-1,))
        output_expect = cpu_output.reshape((-1,))
