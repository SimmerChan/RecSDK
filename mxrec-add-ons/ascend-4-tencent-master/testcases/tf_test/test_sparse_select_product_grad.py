from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from npu_device.compat.v1 import *

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.flags.DEFINE_string("local_log_dir", "output/train_logs.txt", "Log file path")
FLAGS = tf.compat.v1.flags.FLAGS

import os.path
import tensorflow as tf

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

# print("os.path.dirname:", os.path.dirname(__file__))
# filename = os.path.join(os.path.dirname(__file__), 'libsmart_tfdl.so')
# print("filename:", filename)
filename = os.path.join("/root/code/ascend-4-tencent/testcases/tf_test/libsmart_tfdl.so")
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
    grad = tf.Variable([[0.8, 0.7, 1.5, 0.3],
                        [1.5, 0.9, 0.6, 0.3],
                        [0, 0, 0, 0]], dtype=tf.float32)
    index = [0, 0, 1]  # 在算子插件中增加逻辑，手动推一个[0]到开头

    out = ops_module.sparse_select_product_grad(index, grad)
    init = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session(config=config('cpu')) as session:
        session.run(init)
        result_cpu = session.run(out)
    with tf.compat.v1.Session(config=config('npu')) as session:
        session.run(init)
        result_npu = session.run(out)

    print("\nresult_cpu:", result_cpu)
    print("\nresult_npu:", result_npu)
    print("=========================")
    cmp_result = np.allclose(result_npu, result_cpu, atol, rtol)
    print(cmp_result)
    print("=========================")


if __name__ == "__main__":
    tf.compat.v1.app.run()
