#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2024 Huawei Technologies Co., Ltd
import numpy as np
import os
import math

def softmax_grad(grad, src):
    dst = grad * src
    dst = np.sum(dst, axis=-1, keepdims=True)
    dst = (grad - dst) * src
    return dst

def param_attn_layer_grad(dout, softmax_out, query, key, value):
    # Dv and dS
    dv = np.matmul(np.transpose(softmax_out, (0, 2, 1)), dout)
    dS = np.matmul(dout, np.transpose(value, (0, 2, 1))) 
    dS = softmax_grad(dS, softmax_out)/math.sqrt(query.shape[2])
    # Atten 
    dQ = np.matmul(dS, key)
    dK = np.matmul(np.transpose(dS, (0, 2, 1)), query)
    return dQ, dK, dv

def gen_golden_data_simple():

    dout = np.random.uniform(-1, 1,[1024, 1000, 80]).astype(np.float32)
    softmax_out = np.random.uniform(-1, 1,[1024, 1000, 50]).astype(np.float32)
    query = np.random.uniform(-1, 1,[1024, 1000, 80]).astype(np.float32)
    key = np.random.uniform(-1, 1,[1024, 50, 80]).astype(np.float32)
    value = np.random.uniform(-1, 1,[1024, 50, 80]).astype(np.float32)

    grad_query, grad_key, grad_value = param_attn_layer_grad(dout, softmax_out, query, key, value)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    dout.tofile("./input/dout.bin")
    softmax_out.tofile("./input/softmax_out.bin")
    query.tofile("./input/query.bin")
    key.tofile("./input/key.bin")
    value.tofile("./input/value.bin")
    
    grad_query.tofile("./output/golden_grad_query.bin")
    grad_key.tofile("./output/golden_grad_key.bin")
    grad_value.tofile("./output/golden_grad_value.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
