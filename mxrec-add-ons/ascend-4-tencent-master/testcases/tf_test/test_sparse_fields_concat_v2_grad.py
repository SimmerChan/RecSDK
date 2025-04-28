from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
    ref = tf.Variable([[[1, 2, 3, 4], [5, 6, 7, 8],
                        [1, 2, 3, 4], [5, 6, 7, 8]],
                       [[10, 11, 12, 13], [14, 15, 16, 17],
                        [10, 11, 12, 13], [14, 15, 16, 17]]
                       ])
    indices = tf.constant([[1, 1]])
    updates = tf.constant([[-14, -15, -16, -17]])
    
    
    weight_data = [0.1, 0.2, 0.3, 0.4, -4.1, -4.2, -4.3, -4.4,
        0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4,
        -0.9, -1.0, -1.1, -1.2, 0.5, 0.6, 0.7, 0.8,
        1.3, 1.4, 1.5, 1.6, -0.9, -1.0, -1.1, -1.2,
        1.7, 1.8, 1.9, 2.0, 1.3, 1.4, 1.5, 1.6,
        2.1, 2.2, 2.3, 2.4, 1.7, 1.8, 1.9, 2.0,
        -2.5, -2.6, -2.7, -2.8, 2.1, 2.2, 2.3, 2.4,
        2.9, 3.0, 3.1, 3.2, -2.5, -2.6, -2.7, -2.8,
        3.3, 3.4, 3.5, 3.6, 2.9, 3.0, 3.1, 3.2,
        3.7, 3.8, 3.9, 4.0, 3.3, 3.4, 3.5, 3.6,
        -4.1, -4.2, -4.3, -4.4, 3.7, 3.8, 3.9, 4.0,]
    weight = np.array(weight_data).astype(np.float32).reshape(11,2,4)
    
    field_data = [1, 1, 2, 4, 1, 3,
        1, 1, 1, 3, 2, 2, 2, 4, 1, 3,
        2, 2, 2, 2,
        1, 3,]
    field = np.array(field_data).astype(np.int32).reshape(11,2)
    
    index_data = [
      0, 0, 0, 1, 1, 1, 1, 1 ,2, 2, 3, 5
    ]
    index = np.array(index_data).astype(np.int32)
    
    grad_part1 = np.random.uniform(0, 0.1, (5, 16)).astype(np.float32)
    grad_part2 = np.random.uniform(0, 0.1, (5, 16)).astype(np.float32)
    keys_per_fields_data = [1, 0, 1, 1,
        1, 1, 2, 1,
        0, 2, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 0,]
    keys_per_fields = np.array(keys_per_fields_data).astype(np.int32).reshape(5,4)
    
    weight_tensor = tf.convert_to_tensor(weight)
    field_tensor = tf.convert_to_tensor(field)
    index_tensor = tf.convert_to_tensor(index)
    grad_part1_tensor = tf.convert_to_tensor(grad_part1)
    grad_part2_tensor = tf.convert_to_tensor(grad_part2)
    keys_per_fields_tensor = tf.convert_to_tensor(keys_per_fields)


    out = ops_module.sparse_fields_concat_v2_grad(weight_tensor, field_tensor, index_tensor, grad_part1_tensor, grad_part2_tensor, keys_per_fields_tensor, fw_field_num=4, part1_fields=[1,3], part2_fields=[2,4])
    
    #out = tf.compat.v1.scatter_nd_add(ref, indices, updates)
    
    init = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session(config=config('cpu')) as session:
        session.run(init)
        result_cpu = session.run(out)
    with tf.compat.v1.Session(config=config('npu')) as session:
        session.run(init)
        result_npu = session.run(out)
    
    print('====================================')
    print("result_npu:", result_npu, " max:", np.max(result_npu), " min:", np.min(result_npu))
    print("result_cpu:", result_cpu, " max:", np.max(result_cpu), " min:", np.min(result_cpu))
    cmp_result = np.allclose(result_npu, result_cpu, atol, rtol)
    print("compare result: ", cmp_result)
    print('====================================')


if __name__ == "__main__":
    tf.compat.v1.app.run()