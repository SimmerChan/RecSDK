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
import npu_device
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
    input_q_np = np.ones((batch_size, seq, num_heads, attn_dim), dtype=np.float16) / 10
    input_k_np = np.ones((batch_size, seq, num_heads, attn_dim), dtype=np.float16) / 10
    input_v_np = np.ones((batch_size, seq, num_heads, attn_dim), dtype=np.float16) / 100
    input_attn_mask_np = np.tril(np.ones((batch_size, num_heads, seq, seq), dtype=np.float16))
    input_attn_bias_np = np.ones((batch_size, num_heads, seq, seq), dtype=np.float16)
    return input_q_np, input_k_np, input_v_np, input_attn_mask_np, input_attn_bias_np


def get_optional_input_list(input_list):
    result_list = [int(0)]
    if input_list is not None:
        result_list.append(input_list)
    return result_list


def hstu_dense_forward(
    input_q, input_k, input_v, mask, input_attn_bias, input_seq_offsets=None
):
    seq = input_q.shape[2]

    ascend_out = tfOpLib.HstuDenseForward(
        q=input_q,
        k=input_k,
        v=input_v,
        attn_bias=input_attn_bias,
        mask=mask,
        maskType=0,
        max_seq_len=seq,
        silu_scale=0.000977,
        layout="normal",
        seq_offsets=get_optional_input_list(input_seq_offsets),
    )
    return ascend_out


def hstu_dense_forward_gloden(input_q, input_k, input_v, mask, input_attn_bias, silu_scale):
    def silu_gloden(x):
        return x / (1 + np.exp(-1 * x))
    
    q = np.transpose(input_q, (0, 2, 1, 3))
    k = np.transpose(input_k, (0, 2, 3, 1))
    qk_attn = np.matmul(q, k)

    qk_attn = qk_attn.astype(np.float32)
    bias = input_attn_bias.astype(np.float32)
    mask = mask.astype(np.float32)
    qk_attn = qk_attn + bias

    qk_attn = silu_gloden(qk_attn) * silu_scale
    qk_attn = qk_attn * mask

    v = np.transpose(input_v, (0, 2, 1, 3))
    atten_output = np.matmul(qk_attn, v)
    atten_output = np.transpose(atten_output, (0, 2, 1, 3))
    return atten_output.astype(np.float16)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq", [256, 512])
@pytest.mark.parametrize("num_heads", [2, 4])
@pytest.mark.parametrize("attn_dim", [32, 64])
def test_hstu_dense_forward(batch_size, seq, num_heads, attn_dim):
    input_q = tf.placeholder(
        tf.float16, shape=[batch_size, seq, num_heads, attn_dim], name="q"
    )
    input_k = tf.placeholder(
        tf.float16, shape=[batch_size, seq, num_heads, attn_dim], name="k"
    )
    input_v = tf.placeholder(
        tf.float16, shape=[batch_size, seq, num_heads, attn_dim], name="v"
    )
    input_attn_bias = tf.placeholder(
        tf.float16, shape=[batch_size, num_heads, seq, seq], name="attn_bias"
    )
    mask = tf.placeholder(
        tf.float16, shape=[batch_size, num_heads, seq, seq], name="mask"
    )

    ascend_out = hstu_dense_forward(input_q, input_k, input_v, mask, input_attn_bias)

    input_q_np, input_k_np, input_v_np, input_attn_mask_np, input_attn_bias_np = generate_data(
        batch_size, seq, num_heads, attn_dim
    )

    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        attn_out = sess.run(
            ascend_out,
            feed_dict={
                input_q: input_q_np,
                input_k: input_k_np,
                input_v: input_v_np,
                mask: input_attn_mask_np,
                input_attn_bias: input_attn_bias_np
            },
        )

    attn_out_gloden = hstu_dense_forward_gloden(input_q_np,
        input_k_np, input_v_np, input_attn_mask_np, input_attn_bias_np, 0.000977)
    res = np.allclose(attn_out, attn_out_gloden, 1e-3, 1e-3)
    assert res == True
