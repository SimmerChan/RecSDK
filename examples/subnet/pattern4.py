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


@tf.function(jit_compile=True) # 启用 XLA JIT 编译
def process_tensors(tensors_args, clip_min, clip_max):
    # Step 1: 对每个 tensor 应用 tf.clip_by_value
    clipped_tensors = [tf.clip_by_value(t, clip_min, clip_max) for t in tensors_args]

    # Step 2: 使用 tf.concat 将所有 tensor 沿 axis=1 拼接
    output_tensor = tf.concat(clipped_tensors, axis=1)

    return output_tensor


# 创建 201 个形状为 (128, 1) 的随机 tensor
tensors = [tf.random.normal((128, 1)) for _ in range(201)]

# 设置 clip 的最小值和最大值
CLIP_MIN = -1.0
CLIP_MAX = 1.0

# 调用函数处理这些 tensors
output_tensor = process_tensors(tensors, CLIP_MIN, CLIP_MAX)
