from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from dataclasses import fields

# Imports
import logging
import tensorflow as tf
import numpy as np
from npu_device.compat.v1 import *

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.flags.DEFINE_string("local_log_dir", "output/train_logs.txt", "Log file path")
FLAGS = tf.compat.v1.flags.FLAGS

import os.path as osp
import tensorflow as tf

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

filename = osp.join(osp.dirname(__file__), 'libsmart_tfdl.so')
ops_module = tf.load_op_library(filename)
print(dir(ops_module))

atol = 0.001
rtol = 0.001

def config(excute_type):
    if excute_type == 'npu':
        session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["enable_data_pre_proc"].b = True
        custom_op.parameter_map["mix_compile_mode"].b = True
        custom_op.parameter_map["use_off_line"].b = True
        custom_op.parameter_map["min_group_size"].b = 1

    elif excute_type == 'cpu':
        session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)

    return session_config

def main(unused_argv):
    embedding_1 = np.random.uniform(0, 0.1, (10, 4)).astype(np.float32)
    embedding_2 = np.random.uniform(0, 0.1, (10, 8)).astype(np.float32)
    embedding_3 = np.random.uniform(0, 0.1, (10, 2)).astype(np.float32)
    embedding_4 = np.random.uniform(0, 0.1, (10, 16)).astype(np.float32)
    embedding_5 = np.random.uniform(0, 0.1, (10, 16)).astype(np.float32)
    embedding_6 = np.random.uniform(0, 0.1, (10, 1)).astype(np.float32)
    embedding_7 = np.random.uniform(0, 0.1, (10, 8)).astype(np.float32)
    embedding_8 = np.random.uniform(0, 0.1, (10, 6)).astype(np.float32)
    embedding_9 = np.random.uniform(0, 0.1, (10, 4)).astype(np.float32)
    batch_size = np.random.randint(1, 10, size=(9)).astype(np.int32)

    embedding_tensor_1 = tf.convert_to_tensor(embedding_1)
    embedding_tensor_2 = tf.convert_to_tensor(embedding_2)
    embedding_tensor_3 = tf.convert_to_tensor(embedding_3)
    embedding_tensor_4 = tf.convert_to_tensor(embedding_4)
    embedding_tensor_5 = tf.convert_to_tensor(embedding_5)
    embedding_tensor_6 = tf.convert_to_tensor(embedding_6)
    embedding_tensor_7 = tf.convert_to_tensor(embedding_7)
    embedding_tensor_8 = tf.convert_to_tensor(embedding_8)
    embedding_tensor_9 = tf.convert_to_tensor(embedding_9)

    embedding_list = [embedding_tensor_1, embedding_tensor_2, embedding_tensor_3, embedding_tensor_4, embedding_tensor_5, embedding_tensor_6, embedding_tensor_7, embedding_tensor_8, embedding_tensor_9]
    batch_size = tf.convert_to_tensor(batch_size)

    output = ops_module.dense_select_input_v2(embedding_list, batch_size, fields=[101, 112, 123, 134, 145, 156, 167, 178, 189], select_parts=[1, 2, 1, 2, 1, 2, 1], select_fields=[101, 112, 123, 156, 167, 178, 189], embedding_sizes=[4, 8, 2, 1, 8, 6, 4])

    init = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session(config=config('cpu')) as session:
        session.run(init)
        result_cpu = session.run(output)
    with tf.compat.v1.Session(config=config('npu')) as session:
        session.run(init)
        result_npu = session.run(output)

    print('====================================')
    print("result_npu:", result_npu[0], " max:", np.max(result_npu[0]), " min:", np.min(result_npu[0]))
    print("result_cpu:", result_cpu[0], " max:", np.max(result_cpu[0]), " min:", np.min(result_cpu[0]))
    cmp_result = np.allclose(result_npu[0], result_cpu[0], atol, rtol)
    print("compare result [0]: ", cmp_result)
    cmp_result = np.allclose(result_npu[1], result_cpu[1], atol, rtol)
    print("compare result [1]: ", cmp_result)
    cmp_result = np.allclose(result_npu[2], result_cpu[2], atol, rtol)
    print("compare result [2]: ", cmp_result)
    print('====================================')


if __name__ == "__main__":
    tf.compat.v1.app.run()
