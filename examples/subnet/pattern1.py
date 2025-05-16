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


tf.config.optimizer.set_jit(True)

def process_tensor(input_tensor):
    # Step 1: 先 reshape 为 (128, 192, 1, 256)
    reshaped_tensor = tf.reshape(input_tensor, (128, 192, 1, 256))

    # Step 2: BatchNormalization (假设使用标准的 BN 操作)
    # 注意：BatchNormalization 在训练模式下会有不同的行为，因此需要指定训练模式
    batch_norm_tensor = tf.keras.layers.BatchNormalization()(reshaped_tensor, training=False)

    # Step 3: 再次 reshape 为 (128, 192, 2, 256)
    reshaped_bn_tensor = tf.reshape(batch_norm_tensor, (128, 192, 256))

    # Step 4: 创建一个常量张量，形状为 (128, 192, 2, 256)
    constant_tensor = tf.constant(1.0, shape=(128, 192, 256))

    # Step 5: 做逐元素相减操作
    sub_output = tf.subtract(reshaped_bn_tensor, constant_tensor)

    mul_output = tf.multiply()

    # 最后，reshape 为 (128, 192, 256)
    final_result = tf.reshape(sub_output, (128, 192, 256))

    return final_result


# 创建一个示例输入张量，形状为 (128, 192, 256)
input_tensor = tf.random.normal((128, 192, 256))

# 调用函数
output_tensor = process_tensor(input_tensor)

# 打印输出形状
print("Output shape:", output_tensor.shape)  # 输出应为 (128, 192, 256)
