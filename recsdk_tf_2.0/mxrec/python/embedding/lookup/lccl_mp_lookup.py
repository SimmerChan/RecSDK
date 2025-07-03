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
from mxrec.python.config import get_all2all_op_type
from mxrec.python.config.parser import TomlParser
from mxrec.python.communication import get_min_device_id, get_local_rank_size


class LcclMPLookup(MPLookup):
    """Embedding lookup for model parallel, and the hash strategy for the lookup key is modulo."""

    def __init__(self, emb_table: Union[StaticEmbTable], ids: tf.Tensor):
        super(LcclMPLookup, self).__init__(emb_table, ids)
        self.local_rank_size = get_local_rank_size()
        self.all2all_op_type = get_all2all_op_type()
        logger.info("[use_lccl_all2all_op: [true], all2all_op_type: [%s]]", self.all2all_op_type)

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

                local_embedding_grad = self._embedding_all2all(
                    own_embedding_grad, lookup_params.send_count_matrix, "local_embedding_grad", is_bp=True
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

    def _embedding_all2all(
        self, emb: tf.Tensor, sc_matrix: Optional[List[List[tf.Tensor]]], name: str, is_bp: bool = False
    ) -> tf.Tensor:  # pragma: no cover
        """Perform all-to-all communication for embedding.

        For example, in training, 2 ranks has there local embedding:
            rank0 emb: [[3, 3, 3, 3, 3, 3],
                        [3, 3, 3, 3, 3, 3],
                        [7, 7, 7, 7, 7, 7]]
            rank1 emb: [[0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                        [0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
                        [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                        [0.6, 0.6, 0.6, 0.6, 0.6, 0.6]]
        The send count matrix:
            sc matrix: [[1, 2],
                        [2, 2]]
        The embedding all-to-all matrix after transpose:
            embedding all-to-all matrix: [[1 * embedding_dimension, 2 * embedding_dimension],
                                          [2 * embedding_dimension, 2 * embedding_dimension]]
        After all-to-all, each rank get:
            rank0 emb: [[3, 3, 3, 3, 3, 3],
                        [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                        [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]]
            rank1 emb: [[3, 3, 3, 3, 3, 3],
                        [7, 7, 7, 7, 7, 7],
                        [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                        [0.6, 0.6, 0.6, 0.6, 0.6, 0.6]]

        Args:
            emb: The local embedding or embedding grad.
            sc_matrix: The send count matrix.
            name: The name of the 'emb' in the graph after all-to-all communication.
            is_bp: Whether it is backpropagation.

        Returns:
            The 'emb' after all-to-all communication.

        """
        if self._rank_size > 1:
            emb_all2all_matrix = sc_matrix * self._emb_table.dim
            if not is_bp:
                emb_all2all_matrix = tf.transpose(emb_all2all_matrix)

            if name == "local_embedding_grad":
                emb = self.lccl_all2all(emb, sc_matrix, emb_all2all_matrix)
            else:
                emb = hccl_ops.all_to_all_v_c(send_data=emb, send_count_matrix=emb_all2all_matrix, rank=self._rank_id)

        emb = tf.reshape(emb, shape=(-1, self._emb_table.dim), name=name)
        return emb

    def lccl_all2all(self, emb: tf.Tensor, sc_matrix: Optional[List[List[tf.Tensor]]], emb_all2all_matrix) -> tf.Tensor:
        shape_vec = tf.cast(sc_matrix, dtype=tf.int32)
        if self.all2all_op_type == "all2all_v_c":
            return self.lccl_ops_so.all2allvc(
                send_data=emb,
                send_count_matrix=emb_all2all_matrix,
                shape_vec=shape_vec,
                peer_mem=self.peer_mem,
                rank=self._rank_id,
                rank_size=self._rank_size,
                local_rank_size=self.local_rank_size,
                dim=self._emb_table.dim)
        elif self.all2all_op_type == "all2all_v_c_fm":
            return self.lccl_ops_so.all2allvc_fm(
                send_data=emb,
                send_count_matrix=emb_all2all_matrix,
                shape_vec=shape_vec,
                peer_mem=self.peer_mem,
                rank=self._rank_id,
                rank_size=self._rank_size,
                dim=self._emb_table.dim)
        elif self.all2all_op_type == "all2all_v_c_cf":
            return self.lccl_ops_so.all2allvc_cf(
                send_data=emb,
                send_count_matrix=emb_all2all_matrix,
                shape_vec=shape_vec,
                peer_mem=self.peer_mem,
                rank=self._rank_id,
                rank_size=self._rank_size,
                local_rank_size=self.local_rank_size,
                dim=self._emb_table.dim)
        else:
            logger.error("all2all_op_type is wrong, please check")
            raise ValueError("all2all_op_type is wrong, please check")