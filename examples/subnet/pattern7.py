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
input1 = tf.random.normal((128, 150, 148))  # Shape (128, 150, 148)
input2 = tf.random.normal((128, 150, 148))  # Shape (128, 150, 148)
input3 = tf.random.normal((128, 150, 1))


# Enable XLA by wrapping the function with tf.function
@tf.function(jit_compile=True)
def compute_operations(input1_arg, input2_arg, input3_arg):
    add_output = input1_arg + input2_arg
    reduce_output = tf.reduce_mean(add_output, axis=-1, keepdims=True)
    sub_output = add_output - reduce_output
    square_output = tf.square(add_output - reduce_output)
    reduce2_output = tf.reduce_mean(square_output, axis=-1, keepdims=True)
    add2_output = reduce2_output + input3_arg
    rsqrt_output = tf.math.rsqrt(add2_output)
    mul_output = sub_output * rsqrt_output
    const_tensor1 = tf.constant(1.0, shape=(148,))
    const_tensor2 = tf.constant(2.0, shape=(148,))
    mul2_output = mul_output * const_tensor1
    add3_output = mul2_output + const_tensor2
    return add3_output


# Call the function
final_output = compute_operations(input1, input2, input3)

# Print the final output shape
print("Final output shape:", final_output.shape)
