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

import os
from math import sqrt
import numpy as np


def softmax(src):
    # do softmax with numpy
    rowmax = np.max(src, axis=-1, keepdims=True)
    sub = src - rowmax
    exp_sub = np.exp(sub)
    rowsum = np.sum(exp_sub, axis=-1, keepdims=True)
    dst = exp_sub / rowsum
    return dst


def gloden_atten_fusion(query, key, value, atten_mask):
    qk = np.matmul(query, key.transpose(0, 2, 1))
    attn_dimsqrt = 1 / sqrt(query.shape[2])
    attn_weight = np.multiply(qk, attn_dimsqrt)
    atten_mask = np.add(10000, np.multiply(atten_mask, -10000))
    add_mask = np.add(attn_weight, atten_mask)
    qk_div = softmax(add_mask)

    out = np.matmul(qk_div, value)
    return out, qk_div


def gen_golden_data_simple():
    input_query = np.random.uniform(-1, 1, [1024, 1000, 80]).astype(np.float32)
    input_key = np.random.uniform(-1, 1, [1024, 50, 80]).astype(np.float32)
    input_value = np.random.uniform(-1, 1, [1024, 50, 80]).astype(np.float32)
    input_atten_mask = np.random.randint(0, 2, size=(1024, 1000, 50)).astype(np.float32)

    golden_atten_score, gold_softmax_out = gloden_atten_fusion(input_query, input_key, input_value, input_atten_mask)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_query.tofile("./input/input_query.bin")
    input_key.tofile("./input/input_key.bin")
    input_value.tofile("./input/input_value.bin")
    input_atten_mask.tofile("./input/input_atten_mask.bin")
    
    golden_atten_score.tofile("./output/golden_atten_score.bin")
    gold_softmax_out.tofile("./output/golden_softmax_out.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
