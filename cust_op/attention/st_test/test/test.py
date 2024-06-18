import os
import sys
import npu_device
from npu_device.compat.v1.npu_init import *
from mpi4py import MPI
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

os.environ["DEVICE_ID"] = str(0)
os.environ["ASCEND_DEVICE_ID"] = str(0)
os.environ["JOB_ID"] = "10086"

tf.compat.v1.disable_eager_execution()
tfOpLib = tf.load_op_library("../build/tf_ops/libattention_ops.so")

npu_device.compat.enable_v1()
npu_init = npu_ops.initialize_system()
npu_shutdown = npu_ops.shutdown_system()
config = tf.compat.v1.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF


def attention_fusion(q, k, v, mask=None):
    tf_shape = tf.shape(k)
    tf_shape_query = tf.shape(q)

    if mask is None:
        mask = tf.zeros(tf_shape[0], tf_shape_query[1], tf_shape[1])

    attn_out, softmax_out = tfOpLib.attention_fusion(query=q, key=k, value=v, atten_mask=mask, mask_on = 1)
    return attn_out, softmax_out


@ops.RegisterGradient("AttentionFusion")
def _npu_fusion_attention_grad(op, *grad):
    q = op.inputs[0]
    k = op.inputs[1]
    v = op.inputs[2]
    mask = op.inputs[3]

    attention_out = op.outputs[0]
    softmax_output = op.outputs[1]
    dout = grad[0]
    d_q, d_k, d_v = tfOpLib.attention_fusion_grad(dout=dout, softmax_out=softmax_output, query=q, key=k, value=v)
    return d_q, d_k, d_v, tf.zeros(tf.shape(mask))


def param_attn_layer(q, k, v, m=None):
    # q k v 
    with tf.name_scope("param_attn_layer"):
        with tf.name_scope("qk_matmul"):
            qk = tf.matmul(q, k, transpose_b=True)
            
        with tf.name_scope("div"):
            sqrt_attndim = tf.sqrt(tf.cast(tf.shape(k)[2], tf.float32))
            if sqrt_attndim is 0 :
                qk_div = 0
            else:
                qk_div = qk / sqrt_attndim
            if m is not None:
                m = qk_div + (1 - m) * 10000
            else:
                m = qk_div
        with tf.name_scope("softmax_out"):
            softmax_output = tf.nn.softmax(m, axis=-1)

        with tf.name_scope("out_matmul"):    
            out = tf.matmul(softmax_output, v)

    return out, softmax_output
#测试用例


def generate_data(dim0, dim1, dim2, dim3, dim4):
    q = np.random.randn(dim0, dim1, dim2).astype(np.float32)
    k = np.random.randn(dim0, dim3, dim2).astype(np.float32)
    v = np.random.randn(dim0, dim3, dim4).astype(np.float32)
    m = np.zeros((dim0, dim1, dim3)).astype(np.float32)
    return q, k, v, m


query = tf.placeholder(tf.float32, shape=[None, None, None], name="query")
key = tf.placeholder(tf.float32, shape=[None, None, None], name="key")
value = tf.placeholder(tf.float32, shape=[None, None, None], name="value")
mask = tf.placeholder(tf.float32, shape=[None, None, None], name="mask")

# gloden
atten_out_gloden, softmax_out_gloden = param_attn_layer(query, key, value, mask)
loss_golden = tf.reduce_mean(atten_out_gloden, keep_dims=False)
grads_and_vars_golden = tf.gradients(loss_golden, [query, key, value])

# fusion
atten_out, softmax_out = attention_fusion(q=query, k=key, v=value, mask=mask)
loss = tf.reduce_mean(atten_out, keep_dims=False)
grads_and_vars = tf.gradients(loss, [query, key, value])

# test case
test_case = [(1024, 50, 80, 50, 80), (1024, 1000, 80, 50, 80), (1024, 1, 64, 50, 64), (1024, 1, 64, 1000, 80),
             (1024, 144, 64, 1000, 80)]


with tf.compat.v1.Session(config=config) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for dim0, dim1, dim2, dim3, dim4 in test_case:
        print("===================test case ", dim0, dim1, dim2, dim3, dim4, " ===================")
        query_np, key_np, value_np, mask_np = generate_data(dim0, dim1, dim2, dim3, dim4)
        for i in range(10):
            result_gloden = sess.run([loss_golden, grads_and_vars_golden, softmax_out_gloden],
                                     feed_dict={query: query_np, key:key_np, value:value_np, mask:mask_np})
            result = sess.run([loss, grads_and_vars, softmax_out],
                              feed_dict={query: query_np, key:key_np, value:value_np, mask:mask_np})
        print(((result[1][0] - result_gloden[1][0]) < 1e-3).all())
        print(((result[1][1] - result_gloden[1][1]) < 1e-3).all())
        print(((result[1][2] - result_gloden[1][2]) < 1e-3).all())
        print("============ attention fusion end =============")
    





