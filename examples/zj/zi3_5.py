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
import dataclasses

import numpy as np
import tensorflow as tf

# 禁用 TensorFlow 2.x 的急切执行模式
tf.compat.v1.disable_eager_execution()
ops = tf.load_op_library(op_so_path)

@dataclasses
class GraphInput:
    input1: tf.float32
    input2: tf.float32
    input3: tf.float32
    input4: tf.float32
    input5: tf.float32
    input6: tf.float32
    input7: tf.float32

input_args = GraphInput()
def build_graph(input_args):
    slice1 = tf.slice(input_args.input1, [0, 0, 0], [128, 50, 128])
    slice2 = tf.slice(input_args.input2, [0, 0, 0], [128, 50, 32])
    slice3 = tf.slice(input_args.input3, [0, 0, 0], [128, 50, 48])
    slice4 = tf.slice(input_args.input4, [0, 0, 0], [128, 50, 48])
    slice5 = tf.slice(input_args.input5, [0, 0, 0], [128, 50, 48])
    slice6 = tf.slice(input_args.input6, [0, 0, 0], [128, 50, 48])
    slice7 = tf.slice(input_args.input7, [0, 0, 0], [128, 50, 48])

    return tf.concat([slice1, slice2, slice3, slice4, slice5, slice6, slice7], axis=2)


if __name__ == '__main__':
    # 创建输入张量
    input_shape = [128, 50, 128]
    input1 = tf.compat.v1.placeholder(tf.float32, shape=input_shape)
    input2 = tf.compat.v1.placeholder(tf.float32, shape=input_shape)
    input3 = tf.compat.v1.placeholder(tf.float32, shape=input_shape)
    input4 = tf.compat.v1.placeholder(tf.float32, shape=input_shape)
    input5 = tf.compat.v1.placeholder(tf.float32, shape=input_shape)
    input6 = tf.compat.v1.placeholder(tf.float32, shape=input_shape)
    input7 = tf.compat.v1.placeholder(tf.float32, shape=input_shape)

    # 创建会话并运行计算
    with tf.compat.v1.Session() as sess:
        i1 = np.random.triangular(left=100, mode=500, right=1000, size=input_shape)
        i2 = np.random.triangular(left=100, mode=500, right=1000, size=input_shape)
        i3 = np.random.triangular(left=100, mode=500, right=1000, size=input_shape)
        i4 = np.random.triangular(left=100, mode=500, right=1000, size=input_shape)
        i5 = np.random.triangular(left=100, mode=500, right=1000, size=input_shape)
        i6 = np.random.triangular(left=100, mode=500, right=1000, size=input_shape)
        i7 = np.random.triangular(left=100, mode=500, right=1000, size=input_shape)

        slice1 = i1[0:128, 0:50, 0:128]
        slice2 = i2[0:128, 0:50, 0:32]
        slice3 = i3[0:128, 0:50, 0:48]
        slice4 = i4[0:128, 0:50, 0:48]
        slice5 = i5[0:128, 0:50, 0:48]
        slice6 = i6[0:128, 0:50, 0:48]
        slice7 = i7[0:128, 0:50, 0:48]
        cpu_output = np.concatenate(
            [slice1, slice2, slice3, slice4, slice5, slice6, slice7],
            axis=2
        )


        input_args.input1 = input1
        input_args.input2 = input2
        input_args.input3 = input3
        input_args.input4 = input4
        input_args.input5 = input5
        input_args.input6 = input6
        input_args.input7 = input7
        output_graph = build_graph(input_args)
        output_xla = sess.run(
            output_graph, feed_dict={input1: i1, input2: i2, input3: i3, input4: i4,
                                     input5: i5, input6: i6, input7: i7})
        output_xla_real = np.array(output_xla).reshape((-1,))
        output_expect = cpu_output.reshape((-1,))
