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

from typing import Optional, List, Dict, Union

import tensorflow as tf

import mxrec_pybind
from mx_rec.util.initialize import ConfigInitializer
from mx_rec.util.tf_version_adapter import npu_ops
from mx_rec.util.log import logger
from mx_rec.core.asc.swap_args import SwapArgs


def get_restore_vector(config):
    logger.debug('Channel %s_restore_%s was built for getnext', config.get("table_name"), config.get("channel_id"))
    if config.get("is_hbm"):
        if not isinstance(config.get("emb_size"), int) or config.get("emb_size") < 1:
            raise TypeError(f"emb_size must be a int")
        if config.get("emb_size") < 1:
            raise ValueError(f"emb_size is less than 1")
        emb_size = config.get("emb_size")
    else:
        if not isinstance(config.get("ext_emb_size"), int) or config.get("ext_emb_size") < 1:
            raise TypeError("ext_emb_size must be a int")
        if config.get("ext_emb_size") < 1:
            raise ValueError("ext_emb_size is less than 1")
        emb_size = config.get("ext_emb_size")

    if ConfigInitializer.get_instance().use_static:
        restore_size = config.get("batch_size") * config.get("feat_cnt")
    else:
        restore_size = None

    with tf.compat.v1.variable_scope(config.get("table_name"), reuse=tf.compat.v1.AUTO_REUSE):
        device_id = int(config.get("device_id"))
        hot_size = int(mxrec_pybind.get_ub_hot_size(device_id) / emb_size)
        restore_vector, hot_pos = npu_ops.gen_npu_ops.get_next(
            output_types=[tf.int32, tf.int32],
            output_shapes=[restore_size, [hot_size]],
            channel_name=f'{config.get("table_name")}_restore_{config.get("channel_id")}')

    return restore_vector, hot_pos


def get_id_offsets(max_lookup_vec_size, config):
    logger.debug('Channel %s_lookup_%s was built for getnext', config.get("table_name"), config.get("channel_id"))
    # 自动扩容当前只支持HBM模式，默认没有换入换出
    swap_in_pos = []
    swap_out_pos = []
    swap_in_len = 0
    swap_out_len = 0
    with tf.compat.v1.variable_scope(config.get("table_name"), reuse=tf.compat.v1.AUTO_REUSE):
        if config.get("use_dynamic_expansion"):
            [id_offsets] = npu_ops.gen_npu_ops.get_next(
                output_types=[tf.int64],
                output_shapes=[[max_lookup_vec_size]],
                channel_name=f'{config.get("table_name")}_lookup_{config.get("channel_id")}')
            return id_offsets, swap_in_pos, swap_out_pos, swap_in_len, swap_out_len

        [id_offsets] = npu_ops.gen_npu_ops.get_next(
            output_types=[tf.int32],
            output_shapes=[[max_lookup_vec_size]],
            channel_name=f'{config.get("table_name")}_lookup_{config.get("channel_id")}')
        if config.get("is_hbm"):
            return id_offsets, swap_in_pos, swap_out_pos, swap_in_len, swap_out_len
        swap_in_pos, swap_out_pos, swap_in_len, swap_out_len = npu_ops.gen_npu_ops.get_next(
            output_types=[tf.int32, tf.int32, tf.int32, tf.int32],
            output_shapes=[[max_lookup_vec_size], [max_lookup_vec_size], [], []],
            channel_name=f'{config.get("table_name")}_swap_all')
        logger.debug('Channel %s_swap_all was built for getnext', config.get("table_name"))
    return id_offsets, swap_in_pos, swap_out_pos, swap_in_len, swap_out_len


def get_all2all_args(use_static: bool, config: dict) -> Optional[list]:
    """
    Get all2all parameters for dynamic condition
    :param use_static: dynamic or static
    :param config: embedding config
    :return: all2all parametrs
    """
    all2all_args = None
    if use_static:
        return all2all_args

    with tf.compat.v1.variable_scope(config.get("table_name"), reuse=tf.compat.v1.AUTO_REUSE):
        with tf.compat.v1.variable_scope("all2all"):
            logger.debug('Channel %s_a2a_%s was built for getnext', config.get("table_name"), config.get("channel_id"))
            all2all_args = npu_ops.gen_npu_ops.get_next(
                output_types=[tf.int64],
                output_shapes=[[config.get("rank_size"), config.get("rank_size")]],
                channel_name=f'{config.get("table_name")}_all2all_{config.get("channel_id")}',
                name="a2a_get_next")[0] * config.get("emb_size")

    return all2all_args


def get_swap_info(config: Dict[str, Union[str, int]], swap_in_len: int, swap_in_pos: List[tf.Tensor],
                  swap_out_len: int, swap_out_pos: List[tf.Tensor], table: tf.Variable) -> List[tf.Operation]:
    """
    Get swap info if threshold is configured.
    :param config: training job config
    :param swap_in_len: swap in length
    :param swap_in_pos: swap in position
    :param swap_out_len: swap out length
    :param swap_out_pos: swap out position
    :param table: the instance to do swap
    :return: swap info
    """

    # HBM模式
    if config.get("is_hbm"):
        swap_args = SwapArgs()
        swap_args.table_dict[table.name] = {
            'config': config
        }
        return [tf.no_op()]

    use_static = ConfigInitializer.get_instance().use_static
    if use_static:
        swap_out_pos = swap_out_pos[:swap_out_len]
    swap_outs = [tf.gather(one_table, swap_out_pos) for one_table in table]
    swap_out = tf.concat(swap_outs, axis=1)
    logger.debug('Channel %s_d2h_all was built for op outfeed.', config.get("table_name"))
    # emb swap不分train test通道，统一用0
    swap_out_op = npu_ops.outfeed_enqueue_op(
        channel_name=f'{config.get("table_name")}_d2h_all', inputs=[swap_out])

    with tf.control_dependencies([swap_out_op]):
        max_lookup_vec_size = None
        if use_static:
            max_lookup_vec_size = config.get("send_count") * config.get("rank_size")
        with tf.compat.v1.variable_scope("h2d_emb"):
            h2d_emb = npu_ops.gen_npu_ops.get_next(
                output_types=[tf.float32],
                output_shapes=[[max_lookup_vec_size, config.get("ext_emb_size")]],
                channel_name=f'{config.get("table_name")}_h2d_all')[0]
        if use_static:
            h2d_emb = h2d_emb[:swap_in_len, :]
            swap_in_pos = swap_in_pos[:swap_in_len]
        nd_swap_pos = tf.expand_dims(swap_in_pos, 1)
        table_num = len(table)
        h2d_emb_split = tf.split(h2d_emb, table_num, axis=1)
        swap_in_op = [tf.compat.v1.scatter_nd_update(table[i], nd_swap_pos, h2d_emb_split[i])
                      for i in range(table_num)]
        return swap_in_op


def get_preprocessed_tensor_for_asc(table, config):
    use_static = ConfigInitializer.get_instance().use_static
    max_lookup_vec_size = None
    if use_static:
        max_lookup_vec_size = config.get("send_count") * config.get("rank_size")

    with tf.compat.v1.variable_scope("restore_vector"):
        restore_vector, hot_pos = get_restore_vector(config)

    with tf.compat.v1.variable_scope("id_offsets"):
        id_offsets, swap_in_pos, swap_out_pos, swap_in_len, swap_out_len = get_id_offsets(max_lookup_vec_size, config)

    all2all_args = get_all2all_args(use_static, config)

    swap_in = get_swap_info(config, swap_in_len, swap_in_pos, swap_out_len, swap_out_pos, table)

    result = {
        "restore_vector": restore_vector,
        "hot_pos": hot_pos,
        "id_offsets": id_offsets,
        "swap_in_op": swap_in,
        "all2all_args": all2all_args,
    }

    return result
