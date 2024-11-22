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

import argparse
import os
import time

import mxrec_pybind
import numpy as np
import tensorflow as tf
from mpi4py import MPI
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

tf.compat.v1.disable_eager_execution()
comm_pybind = tf.load_op_library("/usr/local/python3.7.5/lib/python3.7/site-packages/mx_rec/libasc/libasc_ops.so")

def set_ascend_env(rank, rank_size, local_rank_size, host, file=None, dev_id=-1, dev_index=1):
    """
    Ascend相关参数
    """
    rank = str(rank)
    rank_size = str(rank_size)
    local_rank_size = int(local_rank_size)
    host = str(host)

    os.environ["MOX_USE_NPU"] = "1"
    os.environ["FUSION_TENSOR_SIZE"] = "2000000000"
    os.environ["MOX_USE_TF_ESTIMATOR"] = "0"
    os.environ["MOX_USE_TDT"] = "1"
    os.environ["HEARTBEAT"] = "1"
    os.environ["CONTINUE_TRAIN"] = "true"

    os.environ["RANK_ID"] = rank
    local_rank_id = int(rank) % int(local_rank_size)
    if dev_id != -1:
        os.environ["DEVICE_ID"] = str(dev_id)
        os.environ["ASCEND_DEVICE_ID"] = str(dev_id)
    else:
        os.environ["DEVICE_ID"] = str(local_rank_id)
        os.environ["ASCEND_DEVICE_ID"] = str(local_rank_id)
    if dev_index != -1:
        os.environ["DEVICE_INDEX"] = str(dev_index)
    else:
        os.environ["DEVICE_INDEX"] = str(local_rank_id)

    os.environ["RANK_SIZE"] = rank_size
    if file:
        os.environ["RANK_TABLE_FILE"] = file
    # no else


    os.environ["HCCL_CONNECT_TIMEOUT"] = "600"

    os.environ["JOB_ID"] = "10086"
    os.environ["SOC_VERSION"] = "Ascend910"
    os.environ["GE_AICPU_FLAG"] = "1"
    os.environ["NEW_GE_FE_ID"] = "1"
    os.environ["EXPERIMENTAL_DYNAMIC_PARTITION"] = "1"
    os.environ["ENABLE_FORCE_V2_CONTROL"] = "1"

class WideDeep:
    def __init__(self, table, lookup_table, matrix):
        self.table = table
        self.lookup = lookup_table
        self.matrix = matrix
        self.forward()

    def forward(self):
        with tf.control_dependencies([self.table, self.lookup]):
            src_2 = comm_pybind.lccl_gather_all(emb_table=self.table,
                                                lookup=self.lookup,
                                                send_count_matrix=self.matrix,
                                                shape_vec=shape_vec,
                                                peer_mem=peer_mem,
                                                rank=rank_id,
                                                rank_size=rank_size,
                                                dim=dim)
            self.op2 = src2
        return self.op2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='base')
    parser.add_argument("--local_rank_size")
    parser.add_argument("--hosts")
    parser.add_argument("--hccl_json")
    args = parser.parse_args()
    local_rank_size = int(args.local_rank_size)

    comm = MPI.COMM_WORLD
    rank_id = comm.Get_rank()
    rank_size = comm.Get_size()
    print(f"rank {rank_id}/{rank_size}")
    local_rank_id = rank_id % rank_size

    peer_mem_ = mxrec_pybind.get_peer_mem(rank_id, rank_size)
    print("python peer_mem_ = ", peer_mem_)
    peer_mem = tf.constant(peer_mem_, dtype=tf.int64)

    set_ascend_env(rank_id, rank_size, local_rank_size, host=args.hosts, file=args.hccl_json)

    # create session
    sess_config = tf.compat.v1.ConfigProto()
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["mix_compile_mode"].b = True
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes('must_keep_origin_dtype')
    sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    custom_op.parameter_map["enable_data_pre_proc"].b = True
    sess_config.gpu_options.allow_growth = True
    custom_op.parameter_map["hcom_parallel"].b = False
    custom_op.parameter_map["op_execute_timeout"].i = 500

    global_start_time = time.time()

    
    dim = 128
    emb_len = 3000000
    look_num = 2048
    emb_table = np.random.randn(emb_len, dim)
    lookup = np.random.randint(0, look_num, size=look_num)

    emb_table = tf.convert_to_tensor(emb_table, dtype=tf.float32)
    lookup = tf.convert_to_tensor(lookup, dtype=tf.int32)

    random_matrix = np.full((8, 8), lookup // 8 * dim)

    random_matrix = tf.convert_to_tensor(random_matrix, dtype=tf.int64)
    peer_mem = tf.convert_to_tensor(peer_mem_, dtype=tf.int64)
    shape_vec = tf.constant([1] * look_num, dtype=tf.int32)
    shape_vec = tf.reshape(shape_vec, [-1, 1])

    # model run parameter
    stop_steps = 100
    # Hybrid end
    tf.compat.v1.disable_eager_execution()

    ######################################
    model = WideDeep(emb_table, lookup, random_matrix)

    with tf.compat.v1.Session(config=sess_config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        print("============start=============")
        # start run loop
        total_start_time = time.time()
        current_steps = 0
        train_finished = False
        while not train_finished:
            try:
                current_steps += 1
                print("current step = ", current_steps)
                #
                run_dict = {
                    "embedding2": model.op2,
                }
                if current_steps == 1:
                    total_start_time = time.time()
                start_time = time.time()
                results = sess.run(fetches=run_dict)

                results1 = np.array(results.get("embedding2"))
                print('1===', results1[:10])
                end_time = time.time()
                print(f"current steps: {current_steps}, step time:{(end_time - start_time) * 1000}")
                if current_steps <= 200:
                    total_start_time = time.time()
                if current_steps >= stop_steps:
                    comm.Barrier()
                    print("train finished 0")
                    train_finished = True
            except tf.errors.OutOfRangeError:
                comm.Barrier()
                print("train finished 1")
                train_finished = True
        MPI.Finalize()
