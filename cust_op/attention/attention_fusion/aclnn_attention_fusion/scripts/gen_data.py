#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import os
from math import sqrt

def softmax(src):
    #基于last轴进行rowmax(按行取最大值)处理
    max = np.max(src, axis=-1, keepdims=True)
    sub = src - max
    exp = np.exp(sub)
    #基于last轴进行rowsum(按行求和)处理
    sum = np.sum(exp, axis=-1, keepdims=True)
    dst = exp / sum
    return dst


def gloden_atten_fusion(query, key, value, atten_mask):
    qk = np.matmul(query, key.transpose(0, 2, 1))
        
    # @jiangli
    print("attn_dim: ", query.shape[2])
    attnDimSqrt = 1 / sqrt(query.shape[2])
    attnWeight = np.multiply(qk, attnDimSqrt)
    atten_mask = np.add(10000, np.multiply(atten_mask, -10000))
    addMask = np.add(attnWeight, atten_mask)
    qk_div = softmax(addMask)

    out = np.matmul(qk_div, value)
    return out, qk_div

def gen_golden_data_simple():
    input_query = np.ones([1024, 1000, 80]).astype(np.float32)
    input_key = np.ones([1024, 50, 80]).astype(np.float32)
    input_key = 0-input_key
    input_value = np.ones([1024, 50, 80]).astype(np.float32)
    input_atten_mask = np.ones([1024, 1000, 50]).astype(np.float32)

    # input_query = np.random.uniform(-1, 1, [1024, 1000, 80]).astype(np.float32)
    # input_key = np.random.uniform(-1, 1, [1024, 50, 80]).astype(np.float32)
    # input_value = np.random.uniform(-1, 1, [1024, 50, 80]).astype(np.float32)
    # input_atten_mask = np.random.randint(0,2,size=(1024, 1000, 50)).astype(np.float32)

    # input_atten_mask = np.random.uniform(-1, 1, [1024, 1000, 50]).astype(np.float32)

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
