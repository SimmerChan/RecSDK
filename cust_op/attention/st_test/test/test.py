from mpi4py import MPI
import os
import numpy as np
from tensorflow.python.framework import ops

os.environ["DEVICE_ID"] = str(0)
os.environ["ASCEND_DEVICE_ID"] = str(0)
os.environ["JOB_ID"] = "10086"

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tfOpLib = tf.load_op_library("../build/tf_ops/libattention_ops.so")
import sys
import npu_device
from npu_device.compat.v1.npu_init import *

npu_device.compat.enable_v1()
npu_init = npu_ops.initialize_system()
npu_shutdown = npu_ops.shutdown_system()
config = tf.compat.v1.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

def attention_fusion(query, key, value, atten_mask=None):
    tf_shape = tf.shape(key)
    tf_shape_query = tf.shape(query)

    # align = ((tf_shape[1]+8-1)//8)*8 - tf_shape[1]

    # padding = tf.zeros((tf_shape[0], align, tf_shape[2]))
    # padding_value = tf.zeros((tf_shape[0], align, tf.shape(value)[2]))
    # padding_mask = tf.ones((tf_shape[0], tf_shape_query[1], align))

    # key = tf.concat((key, padding), axis = 1)
    # value = tf.concat((value, padding_value), axis =1)
    if atten_mask == None:
        atten_mask = tf.zeros(tf_shape[0], tf_shape_query[1], tf_shape[1])
    #atten_mask = tf.concat((atten_mask, padding_mask), axis =2)

    attnOut, softmaxOut = tfOpLib.attention_fusion(query=query, key=key,value=value, atten_mask=atten_mask, mask_on = 1)
    return attnOut, softmaxOut

@ops.RegisterGradient("AttentionFusion")
def _npu_fusion_attention_grad(op, *grad):
    query = op.inputs[0]
    key = op.inputs[1]
    value = op.inputs[2]
    atten_mask = op.inputs[3]

    attention_out = op.outputs[0]
    softmax_out = op.outputs[1]
    dout = grad[0]
    dQuery, dKey, dValue = tfOpLib.attention_fusion_grad(dout=dout, softmax_out=softmax_out, query=query, key=key, value=value)
    return dQuery, dKey, dValue, tf.zeros(tf.shape(atten_mask))

def param_attn_layer(query, key, value, atten_mask=None):
    # q k v 
    with tf.name_scope("param_attn_layer"):
        with tf.name_scope("qk_matmul"):
            qk = tf.matmul(query, key, transpose_b=True)
            
        with tf.name_scope("div"):
            qk_div = qk/tf.sqrt(tf.cast(tf.shape(key)[2], tf.float32))
            if (atten_mask !=None):
                mask = qk_div + (1-atten_mask)*10000
            else:
                mask = qk_div
        with tf.name_scope("softmax_out"):
            softmax_out = tf.nn.softmax(mask, axis=-1)

        with tf.name_scope("out_matmul"):    
            out = tf.matmul(softmax_out, value)

    return out, softmax_out
#测试用例


def generate_data(dim0, dim1, dim2, dim3, dim4):
    query = np.random.randn(dim0, dim1, dim2).astype(np.float32)
    key = np.random.randn(dim0, dim3, dim2).astype(np.float32)
    value = np.random.randn(dim0, dim3, dim4).astype(np.float32)
    mask = np.zeros((dim0, dim1, dim3)).astype(np.float32)
    return query, key, value, mask


query = tf.placeholder(tf.float32, shape=[None, None, None], name="query")
key = tf.placeholder(tf.float32, shape=[None, None, None], name="key")
value = tf.placeholder(tf.float32, shape=[None, None, None], name="value")
mask = tf.placeholder(tf.float32, shape=[None, None, None], name="mask")

# gloden
atten_out_gloden, softmax_out_gloden = param_attn_layer(query, key, value, mask)
loss_golden = tf.reduce_mean(atten_out_gloden, keep_dims=False)
grads_and_vars_golden = tf.gradients(loss_golden, [query, key, value])

# fusion
atten_out, softmax_out = attention_fusion(query=query, key=key, value=value, atten_mask=mask)
loss = tf.reduce_mean(atten_out, keep_dims=False)
grads_and_vars= tf.gradients(loss, [query, key, value])

# test case
test_case = [(1024, 50, 80, 50, 80), (1024, 1000, 80, 50, 80), (1024, 1, 64, 50, 64), (1024, 1, 64, 1000, 80), (1024, 144, 64, 1000, 80)]


with tf.compat.v1.Session(config=config) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for dim0, dim1, dim2, dim3, dim4 in test_case:
        print("===================test case ", dim0, dim1, dim2, dim3, dim4, " ===================")
        query_np, key_np, value_np, mask_np = generate_data(dim0, dim1, dim2, dim3, dim4)
        for i in range(10):
            result_gloden = sess.run([loss_golden, grads_and_vars_golden, softmax_out_gloden], feed_dict={query: query_np, key:key_np, value:value_np, mask:mask_np})
            result = sess.run([loss, grads_and_vars, softmax_out], feed_dict={query: query_np, key:key_np, value:value_np, mask:mask_np})
        print(((result[1][0]-result_gloden[1][0])<1e-3).all())
        print(((result[1][1]-result_gloden[1][1])<1e-3).all())
        print(((result[1][2]-result_gloden[1][2])<1e-3).all())
        print("============ attention fusion end =============")
    





