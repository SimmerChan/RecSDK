#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
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

from typing import Any, List, Optional, Tuple, Union

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_math_ops

from mxrec.python.communication import hccl_ops
from mxrec.python.embedding.lookup.mp_lookup import MPLookup
from mxrec.python.utils import gen_npu_cpu_ops, get_tfa_op, logger
from mxrec.python.constants import MPLookupParams, LOCAL_EMBEDDING_COLLECTION
from mxrec.python.embedding.lookup.base_lookup import BaseLookup
from mxrec.python.embedding.table import StaticEmbTable
from mxrec.python.config import get_fusion_op_type
from mxrec.python.config.parser import TomlParser
from mxrec.python.communication import get_min_device_id


class FusionMPLookup(MPLookup):
    """Embedding lookup for model parallel, and the hash strategy for the lookup key is modulo."""

    def __init__(self, emb_table: Union[StaticEmbTable], ids: tf.Tensor):
        super(FusionMPLookup, self).__init__(emb_table, ids)
        self.fusion_op_type = get_fusion_op_type()
        logger.info("[use_fusion_op: [true], fusion_op_type: [%s]]", self.fusion_op_type)

        import mxrec_pybind
        peer_mem_ = mxrec_pybind.get_peer_mem(self._rank_id, get_min_device_id(), self._rank_size)
        self.peer_mem = tf.constant([peer_mem_[0:self._rank_id]], dtype=tf.int64)
        self.lccl_ops_so = get_tfa_op()

    def lookup(self) -> tf.Tensor:
        """Embedding lookup for model parallel.

        Under model parallelism, the embedding lookup is divided into the following steps:
            Step1: unique and relocate ids.
            Step2: get local embedding.
            Step3: get own embedding.
            Step4: get restore embedding.
            Step5: get sorted embedding.
            Step6: reshape to [bs, seq_len, emb_dim].

        Returns:
            The lookup result.
        """

        @tf.custom_gradient
        def _lookup_forward(embedding: tf.Tensor) -> Any:
            def _lookup_backward(embedding_grad: tf.Tensor) -> ops.IndexedSlices:
                embedding_grad = tf.reshape(embedding_grad, [-1, self._emb_table.dim], name="embedding_grad")

                if self._rank_size > 1:
                    restore_embedding_grad = tf.gather(
                        embedding_grad, lookup_params.sorted_ids_indices, name="restore_embedding_grad"
                    )
                else:
                    restore_embedding_grad = embedding_grad

                own_embedding_grad = tf.compat.v1.unsorted_segment_sum(
                    data=restore_embedding_grad,
                    segment_ids=lookup_params.local_ids_restore,
                    num_segments=tf.shape(own_embedding)[0],
                    name="own_embedding_grad",
                )

                local_embedding_grad = self._embedding_all2all_use_fusion_op(
                    own_embedding_grad, lookup_params, "local_embedding_grad", is_bp=True
                )

                grad = ops.IndexedSlices(
                    values=local_embedding_grad,
                    indices=lookup_params.local_ids,
                    dense_shape=tf.shape(embedding),
                )
                return grad

            # Step3: get own embedding.
            own_embedding = self._embedding_all2all(embedding, lookup_params.send_count_matrix, name="own_embedding")

            # Step4: get restore embedding.
            restore_embedding = tf.gather(own_embedding, lookup_params.local_ids_restore, name="restore_embedding")

            # Step5: get sorted embedding.
            if self._rank_size > 1:
                sorted_embedding = tf.compat.v1.scatter_nd(
                    lookup_params.sorted_ids_indices[:, tf.newaxis],
                    restore_embedding,
                    tf.shape(restore_embedding),
                    name="sorted_embedding",
                )
            else:
                sorted_embedding = restore_embedding

            # Step6: reshape to [bs, seq_len, emb_dim].
            res_shape = tf.concat(
                (tf.shape(self._ids, out_type=self._emb_table.key_dtype), (self._emb_table.dim,)), axis=0
            )
            lookup_res = tf.reshape(sorted_embedding, res_shape, name="lookup_res")

            return lookup_res, _lookup_backward

        with tf.compat.v1.variable_scope(self._get_default_lookup_name()):
            # Step1: unique and relocate ids.
            lookup_params = self._process_ids(self._ids)

            # Step2: get local embedding.
            table_handle = gen_npu_cpu_ops.table_to_resource_v2(table_id=[self._emb_table.table_id])
            local_embedding = gen_npu_cpu_ops.embedding_hash_table_lookup_or_insert(
                table_handle=table_handle,
                keys=lookup_params.local_ids,
                bucket_size=self._emb_table.slice_dev_vocab_size,
                embedding_dim=self._emb_table.dim,
            )
            tf.compat.v1.add_to_collection(LOCAL_EMBEDDING_COLLECTION, local_embedding)
            BaseLookup._local_emb_to_table_ins[local_embedding] = self._emb_table

            return _lookup_forward(local_embedding)

    def _embedding_all2all_use_fusion_op(
            self, own_embedding_grad: tf.Tensor, lookup_params, name: str, is_bp: bool = False
        ) -> tf.Tensor:

        if not self._rank_size > 1:
            local_embedding_grad = tf.reshape(own_embedding_grad, shape=(-1, self._emb_table.dim), name=name)
            return local_embedding_grad
        
        u_local_ids, u_local_idx = tf.unique(lookup_params.local_ids)
        emb_all2all_matrix = lookup_params.send_count_matrix * self._emb_table.dim
        if not is_bp:
            emb_all2all_matrix = tf.transpose(emb_all2all_matrix)
        
        shape_vec = tf.cast(emb_all2all_matrix, dtype=tf.int32)
        if self.fusion_op_type == "all2all_v_c_uss":
            local_embedding_grad = self.lccl_ops_so.all2allvc_uss(
                send_data=own_embedding_grad,
                send_count_matrix=emb_all2all_matrix,
                shape_vec=shape_vec,
                peer_mem=self.peer_mem,
                restore=u_local_idx,
                rank=self._rank_id,
                rank_size=self._rank_size,
                dim=self._emb_table.dim
            )
            local_embedding_grad = tf.reshape(local_embedding_grad, shape=(-1, self._emb_table.dim), name=name)
            return local_embedding_grad
        elif self.fusion_op_type == "all2all_v_c_uss_fm":
            local_embedding_grad = self.lccl_ops_so.all2allvc_uss_fm(
                send_data=own_embedding_grad,
                send_count_matrix=emb_all2all_matrix,
                shape_vec=tf.shape(u_local_ids),
                peer_mem=self.peer_mem,
                restore=u_local_idx,
                rank=self._rank_id,
                rank_size=self._rank_size,
                dim=self._emb_table.dim
            )
            local_embedding_grad = tf.reshape(local_embedding_grad, shape=(-1, self._emb_table.dim), name=name)
            return local_embedding_grad
        elif self.fusion_op_type == "all2all_v_c_uss_cf":
            local_embedding_grad = self.lccl_ops_so.all2allvc_uss_cf(
                send_data=own_embedding_grad,
                send_count_matrix=emb_all2all_matrix,
                shape_vec=tf.shape(u_local_ids),
                peer_mem=self.peer_mem,
                restore=u_local_idx,
                rank=self._rank_id,
                rank_size=self._rank_size,
                dim=self._emb_table.dim
            )
            local_embedding_grad = tf.reshape(local_embedding_grad, shape=(-1, self._emb_table.dim), name=name)
            return local_embedding_grad
        
        else:
            logger.error("fusion_op_type is wrong, please check")
            raise ValueError("fusion_op_type is wrong, please check")