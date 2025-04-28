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


def testcase(weight, fw_weight, field, index):
    weight_tensor = tf.convert_to_tensor(weight)
    fw_weight_tensor = tf.convert_to_tensor(fw_weight)
    field_tensor = tf.convert_to_tensor(field)
    index_tensor = tf.convert_to_tensor(index)

    out = ops_module.sparse_fw_ffm(weight_tensor, fw_weight_tensor, field_tensor,
                                   index_tensor)

    init = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session(config=config('cpu')) as session:
        session.run(init)
        result_cpu = session.run(out)
    with tf.compat.v1.Session(config=config('npu')) as session:
        session.run(init)
        result_npu = session.run(out)

    print('====================================')
    print("result_npu:", result_npu)
    print("result_cpu:", result_cpu)
    cmp_result = np.allclose(result_npu.output, result_cpu.output, atol, rtol)
    print("compare result - output: ", cmp_result)
    cmp_result = np.allclose(result_npu.cross_mean_sum, result_cpu.cross_mean_sum, atol, rtol)
    print("compare result - cross_mean_sum: ", cmp_result)
    cmp_result = np.allclose(result_npu.cross_mean_square_sum, result_cpu.cross_mean_square_sum, atol, rtol)
    print("compare result - cross_mean_square_sum: ", cmp_result)
    cmp_result = np.allclose(result_npu.fw_field_map, result_cpu.fw_field_map, atol, rtol)
    print("compare result - fw_field_map: ", cmp_result)
    print('====================================')


def main(unused_argv):
    weight_data = [
        1., 1., 1., 1.,
        2., 2., 2., 2.,
        3., 3., 3., 3.,
        4., 4., 4., 4.,
        5., 5., 5., 5.,
        6., 6., 6., 6.,
        7., 7., 7., 7.,
        8., 8., 8., 8.,
        9., 9., 9., 9.,
        10., 10., 10., 10.,
        1., 1., 1., 1.,
        2., 2., 2., 2.,
        1., 1., 1., 1.,
        2., 2., 2., 2.,
        3., 3., 3., 3.,
        4., 4., 4., 4.,
        5., 5., 5., 5.,
        6., 6., 6., 6.,
        3., 3., 3., 3.,
        4., 4., 4., 4.,
        5., 5., 5., 5.,
        6., 6., 6., 6.,
        7., 7., 7., 7.,
        8., 8., 8., 8.,
        9., 9., 9., 9.,
        10., 10., 10., 10.,
    ]
    weight = np.array(weight_data).astype(np.float32).reshape(13, 2, 4)
    fw_weight = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32).reshape((6,))
    field = np.array([
        1, 1,
        1, 1,
        1, 1,
        2, 2,
        2, 3,
        2, 2,
        1, 1,
        1, 1,
        1, 1,
        2, 2,
        2, 2,
        2, 2,
        2, 3
    ], dtype=np.int32).reshape((13, 2))
    index = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2]).astype(np.int32)
    
    testcase(weight, fw_weight, field, index)


if __name__ == "__main__":
    tf.compat.v1.app.run()