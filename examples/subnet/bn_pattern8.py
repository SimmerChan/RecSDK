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


# Create the tensors
def batch_norm(x, is_training, decay=0.99, epsilon=1e-3):
    """
    实现 Batch Normalization。

    Args:
        x: 输入张量 (例如：[batch_size, height, width, channels])。
        is_training: 一个布尔值，指示当前是训练阶段还是推理阶段。
        decay: 移动平均的衰减率。
        epsilon: 为了数值稳定性而添加到方差上的一个小常数。

    Returns:
        经过 Batch Normalization 处理后的张量。
    """

    return tf.nn.batch_normalization(x, moving_mean, moving_variance, offset, scale, epsilon)


# Define a custom function that performs all the operations step by step
@tf.function(jit_compile=True)
def compute_operations(input_tensor_arg):
    # Step 1: Reshape input from (128, 192, 256) to (128, 192, 1, 256)
    reshaped_input = tf.reshape(input_tensor_arg, (128, 192, 1, 256))

    # Step 2: Apply Batch Normalization to the reshaped tensor
    batch_norm_tensor = batch_norm(reshaped_input, is_training=False)

    # Step 3: Reshape the batch normalized tensor back to (128, 192, 256)
    reshaped_bn = tf.reshape(batch_norm_tensor, (128, 192, 256))

    # Step 4: Apply Sigmoid activation function
    sigmoid_output = tf.sigmoid(reshaped_bn)

    # Step 5: Create a constant tensor with shape (128, 192, 256)
    const1_tensor = tf.constant(1.0, shape=(128, 192, 256))

    # Step 6: Perform subtraction (sigmoid_output - const_tensor)
    output_tensor = sigmoid_output - const1_tensor

    # Step 7: Perform multiplication (sigmoid_output * input_tensor)
    mul_output = sigmoid_output * input_tensor_arg

    const2_tensor = tf.constant(1.0, shape=(128, 192, 256))
    # Step 8: Perform multiplication of input tensor and constant (input * const_tensor)
    mul2_output = input_tensor_arg * const2_tensor

    # Step 9: Perform multiplication (sub_output * mul2_output)
    mul3_output = output_tensor * mul2_output

    # Step 10: Add mul_output and mul3_output
    add_output = mul_output + mul3_output

    return add_output


# Example input tensor of shape (128, 192, 256)
input_tensor = tf.random.normal((128, 192, 256))
scale = tf.Variable(tf.ones([256]))
offset = tf.Variable(tf.zeros([256]))

# 使用指数移动平均更新全局均值和方差
moving_mean = tf.Variable(tf.zeros([256]), trainable=False)
moving_variance = tf.Variable(tf.ones([256]), trainable=False)

# Call the function to compute the output
final_output = compute_operations(input_tensor)
