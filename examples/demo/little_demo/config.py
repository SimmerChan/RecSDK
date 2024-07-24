# coding: UTF-8
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
import datetime
import math
import os

import tensorflow as tf
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

from mx_rec.util.communication.hccl_ops import get_rank_size

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CURRENT_TIME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
PRECISION_CHECK = bool(int(os.getenv("PRECISION_CHECK", 0)))
PRECISION_CHECK_PATH = SCRIPT_DIR + f'/precision_check/{CURRENT_TIME}'


class Config:
    def __init__(self, mode="simple", task_name="default"):
        self.task_name = task_name
        if mode == "simple":
            self.generate_simple_config()
        else:
            self.generate_large_scale_config()

    def generate_simple_config(self):
        self.batch_number = 8192
        self.batch_size = 4096

        self.key_type = tf.int64
        self.label_type = tf.float32
        self.value_type = tf.float32

        self.item_range = 80000 * get_rank_size()
        self.user_range = 200000 * get_rank_size()
        self.category_range = 5000 * get_rank_size()
        self.item_feat_cnt = 16
        self.user_feat_cnt = 8
        self.category_feat_cnt = 3
        self.access_threshold = 2
        self.eviction_threshold = 2

        rank_size = get_rank_size()
        coefficient = 1.1
        if rank_size != 0:
            max_ui_send_cnt = max(self.item_feat_cnt, self.user_feat_cnt)
            max_ui_range = max(self.item_range, self.user_range)
            self.item_send_cnt = min(int(self.batch_size * self.item_feat_cnt * coefficient),
                                     math.ceil(self.item_range / rank_size))
            self.item_vocab_size = max(self.item_send_cnt * rank_size * rank_size, self.item_range)
            self.user_send_cnt = min(int(self.batch_size * max_ui_send_cnt * coefficient),
                                     math.ceil(max_ui_range / rank_size))
            self.user_vocab_size = max(self.user_send_cnt * rank_size * rank_size, self.user_range)
            self.category_send_cnt = min(int(self.batch_size * self.category_feat_cnt * coefficient),
                                         math.ceil(self.category_range / rank_size))
        else:
            raise ZeroDivisionError("rank size must be an integer which is greater value zero.")

        self.user_hashtable_dim = 32
        self.user_hashtable_threshold = 1
        self.item_hashtable_dim = 8
        self.item_hashtable_threshold = 1

        self.learning_rate = 0.01

    def generate_large_scale_config(self):
        self.lookup_count = 40
        self.tensor_name_list = ["sparse_tensor_%d" % i for i in range(self.lookup_count)]
        self.hashtable_name_list = ["hashtable_%d" % i for i in range(self.lookup_count)]
        self.batch_size = 9600

        self.key_type = tf.int64
        self.label_type = tf.float32
        self.value_type = tf.float32

        self.vocabulary_size = 500000
        self.feat_cnt = 1

        rank_size = get_rank_size()
        coefficient = 1.1
        if rank_size != 0:
            self.send_cnt = min(int(self.batch_size * self.feat_cnt * coefficient),
                                math.ceil(self.vocabulary_size / rank_size))
        else:
            raise ZeroDivisionError("rank size must be an integer which is greater value zero.")

        self.hashtable_dim = 8
        self.learning_rate = 0.01

def construct_basic_config(npu_custom_op):
    npu_custom_op.parameter_map["mix_compile_mode"].b = False
    npu_custom_op.parameter_map["use_off_line"].b = True
    npu_custom_op.parameter_map["min_group_size"].b = 1
    npu_custom_op.parameter_map["HCCL_algorithm"].s = tf.compat.as_bytes("level0:pairwise;level1:pairwise")
    npu_custom_op.parameter_map["enable_data_pre_proc"].b = True
    npu_custom_op.parameter_map["iterations_per_loop"].i = 1
    npu_custom_op.parameter_map["hcom_parallel"].b = False
    npu_custom_op.parameter_map["op_precision_mode"].s = tf.compat.as_bytes("op_impl_mode.ini")
    npu_custom_op.parameter_map["op_execute_timeout"].i = 2000
    npu_custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")


def construct_deterministic_config(npu_custom_op):
    npu_custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("must_keep_origin_dtype")
    npu_custom_op.parameter_map["deterministic"].i = 1

def construct_op_dump_config(npu_custom_op):
    npu_custom_op.parameter_map["enable_dump"].b = True


    dump_path = PRECISION_CHECK_PATH + "/03dump_op"
    os.makedirs(dump_path, exist_ok=True)

    npu_custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(dump_path)
    npu_custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes("0|1|2")
    npu_custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")

    # hot_pos_gather = "user_table//user_table_lookup/hot_pos " \
    #            + "item_table//item_table_lookup/hot_pos"
    gather_for_id_offset = "user_table//user_table_lookup/gather_for_id_offsets " \
                         + "item_table//item_table_lookup/gather_for_id_offsets"
    gather_for_restore_vec = "user_table//user_table_lookup/gather_for_restore_vector " \
                           + "item_table//item_table_lookup/gather_for_restore_vector"
    # gradientUpdate_ScatterNdAdd = "gradients_1/user_table//user_table_lookup/IdentityN_grad/TensorScatterAdd/ScatterNdAdd " \
    #         + "gradients_1/item_table//item_table_lookup/IdentityN_grad/TensorScatterAdd/ScatterNdAdd"
    gradientUpdate_ScatterNdAdd = "LazyAdam_0/update_user_table/ScatterNdAdd " \
                                + "LazyAdam_0/update_user_table/ScatterNdAdd_1 " \
                                + "LazyAdam_0/update_user_table/ScatterNdAdd_2 " \
                                + "LazyAdam_0/update_item_table/ScatterNdAdd " \
                                + "LazyAdam_0/update_item_table/ScatterNdAdd_1 " \
                                + "LazyAdam_0/update_item_table/ScatterNdAdd_2 " \
    
    # lazyadam_byaddress = "LazyAdamByAddress_0/update_user_table//user_table_lookup/id_offsets/user_table/GetNext/EmbeddingLookupByAddress " \
    #                 + "LazyAdamByAddress_0/update_item_table//item_table_lookup/id_offsets/item_table/GetNext/EmbeddingLookupByAddress"
    lookup_byaddress = "user_table//user_table_lookup/EmbeddingLookupByAddress " \
                    + "user_table//item_table_lookup/EmbeddingLookupByAddress"
    update_byaddress = "LazyAdamByAddress_0/update_user_table//user_table_lookup/id_offsets/user_table/GetNext/EmbeddingUpdateByAddress " \
                    + "LazyAdamByAddress_0/update_item_table//item_table_lookup/id_offsets/item_table/GetNext/EmbeddingUpdateByAddress"
    
    use_dynamic_expansion = bool(int(os.getenv("USE_DYNAMIC_EXPANSION", 0)))
    if use_dynamic_expansion:
        op_name_list = [lookup_byaddress, update_byaddress]
    else:
        op_name_list = [gather_for_id_offset, gather_for_restore_vec, gradientUpdate_ScatterNdAdd]

    dump_ops_string = " ".join(op_name_list)
    npu_custom_op.parameter_map["dump_layer"].s = tf.compat.as_bytes(dump_ops_string)

def construct_npu_sess_config(dump_data=False, use_deterministic=0):
    session_config = tf.compat.v1.ConfigProto(allow_soft_placement=False,
                                              log_device_placement=False)
    
    session_config.gpu_options.allow_growth = True
    custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    construct_basic_config(custom_op)
    if use_deterministic:
        construct_deterministic_config(custom_op)
    if dump_data:
        construct_op_dump_config(custom_op)

    session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    session_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

    return session_config
