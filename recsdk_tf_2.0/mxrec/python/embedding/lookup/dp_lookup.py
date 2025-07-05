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
from mxrec.python.utils import gen_npu_cpu_ops
from mxrec.python.constants import DPLookupParams, LOCAL_EMBEDDING_COLLECTION
from mxrec.python.embedding.lookup.base_lookup import BaseLookup
from mxrec.python.embedding.table import StaticEmbTable


class DPLookup(BaseLookup):
    """Embedding lookup for data parallel, and the hash strategy for the lookup key is modulo."""

    def __init__(self, emb_table: StaticEmbTable, ids: tf.Tensor):
        super(DPLookup, self).__init__(emb_table, ids)

    def lookup(self) -> tf.Tensor:
        """Embedding lookup for data parallel.

        For data parallelism, The embedding lookup is divided into two parts:
        1. Get local embedding on each rank
        2. Preprocess keys for backward calculation(see _process_ids for details)

        Returns:
            The lookup result.
        """

        @tf.custom_gradient
        def _lookup_forward(embedding: tf.Tensor) -> Any:
            def _lookup_backward(embedding_grad: tf.Tensor) -> ops.IndexedSlices:
                embedding_grad = tf.reshape(embedding_grad, [-1, self._emb_table.dim], name="embedding_grad")

                # restore_embedding_grad shape [bs*seq_len, emb_dim]
                if self._rank_size > 1:
                    restore_embedding_grad = tf.gather(
                        embedding_grad, lookup_params.rank_idx, name="restore_embedding_grad"
                    )
                else:
                    restore_embedding_grad = embedding_grad

                # local_embedding_grad shape [?, emb_dim]
                local_embedding_grad = self._embedding_all2all(
                    restore_embedding_grad, lookup_params.ids_sc_all, "local_embedding_grad", is_bp=True
                )

                # local_unique_embedding_grad shape [?, emb_dim]
                local_unique_embedding_grad = tf.compat.v1.unsorted_segment_sum(
                    data=local_embedding_grad,
                    segment_ids=lookup_params.local_unique_idx,
                    num_segments=tf.shape(lookup_params.local_unique_ids)[0],
                    name="own_embedding_grad",
                )

                # global_unique_embedding_grad [?, emb_dim]
                if self._rank_size > 1:
                    local_unique_embedding_grad_flatten = tf.reshape(local_unique_embedding_grad, shape=(-1,))
                    local_unique_embedding_grad_flatten_padding = tf.pad(
                        local_unique_embedding_grad_flatten,
                        paddings=[
                            [
                                0,
                                (lookup_params.max_uni_ids_count - lookup_params.local_uni_ids_count)
                                * self._emb_table.dim,
                            ]
                        ],
                        mode="CONSTANT",
                        constant_values=0,
                    )

                    global_unique_embedding_grad = hccl_ops.allgather(
                        local_unique_embedding_grad_flatten_padding, self._rank_size
                    )

                    global_unique_embedding_grad = tf.reshape(
                        global_unique_embedding_grad,
                        shape=(-1, self._emb_table.dim),
                        name="reshape_global_unique_embedding_grad",
                    )

                    global_unique_embedding_grad = tf.gather(
                        global_unique_embedding_grad,
                        lookup_params.global_unique_idx,
                        name="restore_global_unique_embedding_grad",
                        axis=0,
                    )

                else:
                    global_unique_embedding_grad = local_unique_embedding_grad

                grad = ops.IndexedSlices(
                    values=global_unique_embedding_grad,
                    indices=lookup_params.global_unique_ids,
                    dense_shape=tf.shape(embedding),
                )
                return grad
            
            # NOTE:Add unique and restore to avoid the Embedding TableToResourceV2 issue in the vAscend environment. 
            # In the actual hardware environment, delete the following five lines of code:
            ids = tf.reshape(self._ids, shape=(-1,))
            u_local_ids, u_local_idx = tf.unique(ids)
            local_embedding = tf.gather(embedding, u_local_ids)
            restore_embedding = tf.gather(local_embedding, u_local_idx)
            embedding = restore_embedding

            # Step3: reshape to [bs, seq_len, emb_dim].
            res_shape = tf.concat(
                (tf.shape(self._ids, out_type=self._emb_table.key_dtype), (self._emb_table.dim,)), axis=0
            )
            lookup_res = tf.reshape(embedding, res_shape, name="lookup_res")

            return lookup_res, _lookup_backward

        with tf.compat.v1.variable_scope(self._get_default_lookup_name()):
            # Step1: relocate and deduplicate ids to get global unique ids.
            ids = tf.reshape(self._ids, shape=(-1,))
            lookup_params = self._process_ids(ids)

            # Step2: get local embedding.
            table_handle = gen_npu_cpu_ops.table_to_resource_v2(table_id=[self._emb_table.table_id])
            local_embedding = gen_npu_cpu_ops.embedding_hash_table_lookup_or_insert(
                table_handle=table_handle,
                keys=ids,
                bucket_size=self._emb_table.slice_dev_vocab_size,
                embedding_dim=self._emb_table.dim,
            )
            tf.compat.v1.add_to_collection(LOCAL_EMBEDDING_COLLECTION, local_embedding)
            BaseLookup._local_emb_to_table_ins[local_embedding] = self._emb_table
            return _lookup_forward(local_embedding)

    def _process_ids(self, ids: tf.Tensor) -> DPLookupParams:  # pragma: no cover
        """Deduplicate and relocate the feature ids.

        For example, in beginning, 2 ranks has there feature ids:
            rank0: 1, 2, 1, 3
            rank1: 2, 6, 1, 5
        After sort(Reorder the keys, the keys of rank 0 are placed in the front, and the keys of rank 1 are placed
        behind), each rank get:
            rank0: 2, 1, 1, 3
            rank1: 2, 6, 1, 5
        After relocation, each rank get:
            rank0: 2, 2, 6
            rank1: 1, 1, 3, 1, 5
        After unique, each rank get:
            rank0: 2, 6
            rank1: 1, 3, 5
        After Allgather, each rank get:
            rank0: 2, 6, 1, 3, 5
            rank1: 2, 6, 1, 3, 5

        Args:
            ids: feature ids.

        Returns:
            A dataclass for the lookup parameters.

        """

        if self._rank_size == 1:
            # u_ids shape [bs*seq_len*unique_rate,]
            u_ids, u_idx = tf.unique(ids)

            lookup_params = DPLookupParams(
                global_unique_ids=u_ids,
                local_unique_idx=u_idx,
                local_unique_ids=u_ids,
            )
            return lookup_params

        # Reorder the ids. For example, if there are a total of 2 ranks, the ids of rank 0 are placed in the front,
        # and the ids of rank 1 are placed behind. This operation facilitates ALL2ALL to send and receive data.
        mask = tf.cast(tf.math.mod(ids, self._rank_size), tf.int32)  # [bs*seq_len,]
        sorted_indices = tf.argsort(mask)  # [bs*seq_len,]
        sorted_ids = tf.gather(ids, sorted_indices)  # [bs*seq_len,]

        # Relocation ids.
        # send_count shape [rank_size,]
        send_count = gen_math_ops.bincount(
            tf.cast(tf.math.mod(sorted_ids, self._rank_size), tf.int32), self._rank_size, tf.constant([], tf.int64)
        )
        sc_all = hccl_ops.allgather(send_count, self._rank_size)  # [rank_size*rank_size,]
        local_ids = hccl_ops.all_to_all_v_c(send_data=sorted_ids, send_count_matrix=sc_all, rank=self._rank_id)

        # Unique local ids.
        local_unique_ids, local_unique_idx = tf.unique(local_ids)

        # Padding local_unique_ids on each rank to the certain length(max_uni_ids_count) and allgather them.
        local_uni_ids_count = tf.size(local_unique_ids)
        local_uni_ids_count_1d = tf.expand_dims(local_uni_ids_count, axis=0)
        local_uni_ids_count_all = hccl_ops.allgather(local_uni_ids_count_1d, self._rank_size)
        max_uni_ids_count = tf.reduce_max(local_uni_ids_count_all)
        padding_local_unique_ids = tf.pad(
            local_unique_ids,
            paddings=[[0, max_uni_ids_count - local_uni_ids_count]],
            mode="CONSTANT",
            constant_values=-1,
        )
        padding_global_unique_ids = hccl_ops.allgather(padding_local_unique_ids, self._rank_size)

        # Restore the global unique ids
        padding_mask = tf.not_equal(padding_global_unique_ids, -1)
        global_unique_ids = tf.boolean_mask(padding_global_unique_ids, padding_mask)
        global_unique_idx = tf.where(padding_mask)
        global_unique_idx = tf.reshape(global_unique_idx, [-1])

        lookup_params = DPLookupParams(
            rank_idx=sorted_indices,
            ids_sc_all=sc_all,
            local_unique_ids=local_unique_ids,
            local_unique_idx=local_unique_idx,
            global_unique_ids=global_unique_ids,
            global_unique_idx=global_unique_idx,
            max_uni_ids_count=max_uni_ids_count,
            local_uni_ids_count=local_uni_ids_count,
        )

        return lookup_params

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
            emb = hccl_ops.all_to_all_v_c(send_data=emb, send_count_matrix=emb_all2all_matrix, rank=self._rank_id)

        emb = tf.reshape(emb, shape=(-1, self._emb_table.dim), name=name)
        return emb
