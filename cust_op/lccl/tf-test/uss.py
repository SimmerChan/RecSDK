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
    def __init__(self, input_data, matrix, arr, restore):
        self.lbl_hldr = input_data
        self.matrix = matrix
        self.arr = arr
        self.restore = restore
        self.forward()

    def forward(self):
        with tf.control_dependencies([self.lbl_hldr]):
            all2all_result_ = comm_pybind.lccl_all_to_all(send_data=self.lbl_hldr,
                                                          send_count_matrix=self.matrix,
                                                          shape_vec=shape_vec,
                                                          peer_mem=peer_mem,
                                                          rank=rank_id,
                                                          rank_size=rank_size,
                                                          dim=dim)
            all2all_result = tf.reshape(all2all_result_, [-1, dim])
            alluss_result_ = comm_pybind.lccl_all_uss(send_data=self.lbl_hldr,
                                                      send_count_matrix=self.matrix,
                                                      shape_vec=self.arr,
                                                      peer_mem=peer_mem,
                                                      restore=self.restore,
                                                      rank=rank_id,
                                                      rank_size=rank_size,
                                                      dim=dim
                                                      )
            alluss_result = tf.reshape(alluss_result_, [-1, dim])
            self.all2all_result = all2all_result[0]
            self.alluss_result = alluss_result[0]
        return self.all2all_result, self.alluss_result


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
    emb_len = 2048 * 512
    output_len = emb_len // 2
    restore = np.random.randint(0, output_len, size=emb_len)
    arr = np.random.randint(0, output_len, size=emb_len // 2)

    random_matrix = np.full((8, 8), emb_len // 8 * dim)

    send_count = 0
    for i in range(rank_size):
        send_count += int(random_matrix[local_rank_id][i])

    rev_count = 0
    for i in range(rank_size):
        rev_count += int(random_matrix[i][local_rank_id])

    random_send_data = np.random.rand(send_count, 1).astype(np.float32).reshape(-1, dim)

    restore = tf.convert_to_tensor(restore, dtype=tf.int32)
    arr = tf.convert_to_tensor(arr, dtype=tf.int32)
    random_send_data = tf.convert_to_tensor(random_send_data, dtype=tf.float32)
    random_matrix = tf.convert_to_tensor(random_matrix, dtype=tf.int64)
    peer_mem = tf.convert_to_tensor(peer_mem_, dtype=tf.int64)
    shape_vec = tf.constant([1] * (rev_count // dim), dtype=tf.int32)
    shape_vec = tf.convert_to_tensor(shape_vec, dtype=tf.int32)
    shape_vec = tf.reshape(shape_vec, [-1, 1])

    # model run parameter
    stop_steps = 100
    # Hybrid end
    tf.compat.v1.disable_eager_execution()
    ######################################
    model = WideDeep(random_send_data, random_matrix, arr, restore)

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
                    "all2all_result": model.all2all_result,
                    "uss_result": model.alluss_result,
                }
                if current_steps == 1:
                    total_start_time = time.time()
                start_time = time.time()
                results = sess.run(fetches=run_dict)
                print("all2all_result: ", np.array(results.get("all2all_result"))[:3])
                print("uss_result: ", np.array(results.get("uss_result"))[:3])

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
