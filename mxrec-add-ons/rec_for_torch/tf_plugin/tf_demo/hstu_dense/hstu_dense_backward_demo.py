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
import logging
import pytest

import numpy as np
import tensorflow as tf
from npu_device.compat.v1.npu_init import *

logging.getLogger().setLevel(logging.INFO)

device_id = 0

os.environ["DEVICE_ID"] = str(device_id)
os.environ["ASCEND_DEVICE_ID"] = str(0)
os.environ["JOB_ID"] = "10086"

tf.compat.v1.disable_eager_execution()
tfOpLib = tf.load_op_library("../../../tf_ops/build/libHstuDenseOps.so")

config = tf.compat.v1.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["jit_compile"].s = tf.compat.as_bytes("false")
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

def generate_data(batch_size, seq, num_heads, attn_dim):
    input_grad_np = (np.ones((batch_size, seq, num_heads, attn_dim), dtype=np.float16) / 100)
    input_q_np = np.ones((batch_size, seq, num_heads, attn_dim), dtype=np.float16) / 10
    input_k_np = np.ones((batch_size, seq, num_heads, attn_dim), dtype=np.float16) / 10
    input_v_np = np.ones((batch_size, seq, num_heads, attn_dim), dtype=np.float16) / 100
    input_attn_mask_np = np.tril(np.ones((batch_size, num_heads, seq, seq), dtype=np.float16))
    input_attn_bias_np = np.ones((batch_size, num_heads, seq, seq), dtype=np.float16)
    return input_grad_np, input_q_np, input_k_np, input_v_np, input_attn_mask_np, input_attn_bias_np


def get_optional_input_list(input_list):
    result_list = [int(0)]
    if input_list is not None:
        result_list.append(input_list)
    return result_list


def HstuDenseBackward(input_grad, input_q, input_k, input_v, mask, input_attn_bias):
    seq = mask.shape[2]

    ascend_out = tfOpLib.HstuDenseBackward(
        grad=input_grad,
        q=input_q,
        k=input_k,
        v=input_v,
        mask=mask,
        attn_bias=input_attn_bias,
        layout="normal",
        mask_type=0,
        max_seq_len=seq,
        silu_scale=0.000977,
        seq_offsets=get_optional_input_list(None),
    )
    return ascend_out


def HstuDenseBackwardGloden(input_grad, input_q, input_k, input_v, mask, input_attn_bias, silu_scale):
    def sigmoid_gloden(x):
        return 1 / (1 + np.exp(-1 * x))
    
    def silu_gloden(x):
        return x * sigmoid_gloden(x)
    
    qk = np.matmul(np.transpose(input_q, (0, 2, 1, 3)), np.transpose(input_k, (0, 2, 3, 1)))
    gv = np.matmul(np.transpose(input_grad, (0, 2, 1, 3)), np.transpose(input_v, (0, 2, 3, 1)))

    qk = qk.astype(np.float32)
    gv = gv.astype(np.float32)

    mask = mask.astype(np.float32)
    bias = input_attn_bias.astype(np.float32)

    qk = qk + bias
    score = silu_gloden(qk) * silu_scale * mask
    score = score.astype(np.float16)

    v_grad = np.matmul(np.transpose(score, (0, 1, 3, 2)), np.transpose(input_grad, (0, 2, 1, 3)))
    v_grad = np.transpose(v_grad, (0, 2, 1, 3))

    attn_bias_grad = gv * silu_scale * mask * sigmoid_gloden(qk) * (1 + qk * (1 - sigmoid_gloden(qk)))
    attn_bias_grad.astype(np.float16)

    k_grad = np.matmul(np.transpose(attn_bias_grad, (0, 1, 3, 2)), np.transpose(input_q, (0, 2, 1, 3)))
    k_grad = np.transpose(k_grad, (0, 2, 1, 3))

    q_grad = np.matmul(attn_bias_grad, np.transpose(input_k, (0, 2, 1, 3)))
    q_grad = np.transpose(q_grad, (0, 2, 1, 3))

    return q_grad, k_grad, v_grad, attn_bias_grad


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq", [256])
@pytest.mark.parametrize("num_heads", [2, 4])
@pytest.mark.parametrize("attn_dim", [32, 64])
def test_hstu_dense_backward(batch_size, seq, num_heads, attn_dim):
    input_grad = tf.placeholder(tf.float16, shape=[batch_size, seq, num_heads, attn_dim], name="grad")
    input_q = tf.placeholder(tf.float16, shape=[batch_size, seq, num_heads, attn_dim], name="q")
    input_k = tf.placeholder(tf.float16, shape=[batch_size, seq, num_heads, attn_dim], name="k")
    input_v = tf.placeholder(tf.float16, shape=[batch_size, seq, num_heads, attn_dim], name="v")
    input_attn_bias = tf.placeholder(tf.float16, shape=[batch_size, num_heads, seq, seq], name="attn_bias")
    mask = tf.placeholder(tf.float16, shape=[batch_size, num_heads, seq, seq], name="mask")

    ascend_out = HstuDenseBackward(input_grad, input_q, input_k, input_v, mask, input_attn_bias)

    input_grad_np, input_q_np, input_k_np, input_v_np, input_attn_mask_np, input_attn_bias_np = generate_data(
        batch_size, seq, num_heads, attn_dim
    )

    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        q_grad, k_grad, v_grad, attn_bias_grad = sess.run(
            ascend_out,
            feed_dict={
                input_grad: input_grad_np,
                input_q: input_q_np,
                input_k: input_k_np,
                input_v: input_v_np,
                mask: input_attn_mask_np,
                input_attn_bias: input_attn_bias_np
            },
        )

    q_grad_golden, k_grad_golden, v_grad_golden, attn_bias_grad_golden = HstuDenseBackwardGloden(
        input_grad_np, input_q_np, input_k_np, input_v_np, input_attn_mask_np, input_attn_bias_np, 0.000977
    )

    res = np.allclose(q_grad, q_grad_golden, 1e-3, 1e-3)
    assert res == True

    res = np.allclose(k_grad, k_grad_golden, 1e-3, 1e-3)
    assert res == True

    res = np.allclose(v_grad, v_grad_golden, 1e-3, 1e-3)
    assert res == True

    res = np.allclose(attn_bias_grad, attn_bias_grad_golden, 1e-3, 1e-3)
    assert res == True
