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

import collections
from typing import Any, Dict, List, Optional

import tensorflow as tf
from tensorflow.python.ops.init_ops import Initializer as InitializerV1
from tensorflow.python.ops.init_ops_v2 import Initializer as InitializerV2

from mx_rec.core.emb.mergeable_sparse_embedding import MergeableSparseEmbedding
from mx_rec.core.emb.sparse_embedding import SparseEmbedding
from mx_rec.util.singleton import singleton
from mx_rec.util.log import logger


class UnionKey(collections.namedtuple(typename="UnionKey", field_names=["key_dtype", "emb_dim", "initializer_type"])):
    def validate(self) -> None:
        key_dtype = self.key_dtype
        if key_dtype not in (tf.int32, tf.int64):
            raise ValueError("invalid key data type => '{}', expected 'tf.int32' or 'tf.int64'".format(key_dtype))

        emb_dim = self.emb_dim
        if not isinstance(emb_dim, int):
            raise TypeError("invalid embedding dimension type => '{}', expected an 'int'".format(emb_dim))

        initializer_type = self.initializer_type
        if tf.__version__.startswith("1") and not issubclass(initializer_type, InitializerV1):
            raise TypeError("invalid initializer type => '{}', expected an 'tf.Initializer'".format(initializer_type))
        if tf.__version__.startswith("2") and not (
            issubclass(initializer_type, InitializerV1) or issubclass(initializer_type, InitializerV2)
        ):
            raise TypeError("invalid initializer type => '{}', expected an 'tf.Initializer'".format(initializer_type))


@singleton
class MergeableEmbeddingTableProxy:
    def __init__(self) -> None:
        self._mtables: List[MergeableSparseEmbedding] = []
        self._ukey_to_mtable: Dict[UnionKey, SparseEmbedding] = {}
        self._stable_to_mtable: Dict[str, SparseEmbedding] = {}

    def create_mergeable_table(
        self, union_key: UnionKey, small_table_name: str, config: Dict[str, Any]
    ) -> SparseEmbedding:
        union_key.validate()

        mergeable_table = MergeableSparseEmbedding(small_table_name, config=config)
        self._mtables.append(mergeable_table)

        self._ukey_to_mtable[union_key] = mergeable_table
        self._stable_to_mtable[small_table_name] = mergeable_table

        logger.info(
            "A new mergeable embedding table '%s' has been created from embedding table '%s'.",
            mergeable_table.table_name,
            small_table_name,
        )

        return mergeable_table

    def find_mergeable_table(
        self, union_key: Optional[UnionKey], small_table_name: str = ""
    ) -> Optional[SparseEmbedding]:
        union_key.validate()

        if union_key and small_table_name:
            raise ValueError(
                "at most one of union key and table name should be provided, got union key => '{}' and table name => '{}'".format(
                    union_key, small_table_name
                )
            )

        if not (union_key or small_table_name):
            raise ValueError(
                "at least one of union key and table name should be provided, got union key => '{}' and table name => '{}'".format(
                    union_key, small_table_name
                )
            )

        if not union_key:
            return self._stable_to_mtable.get(small_table_name, None)

        return self._ukey_to_mtable.get(union_key, None)

    def join_mergeable_table(
        self, mergeable_table: MergeableSparseEmbedding, small_table_name: str, config: Dict[str, Any]
    ) -> MergeableSparseEmbedding:
        if small_table_name in mergeable_table.merged_small_tables:
            raise ValueError(
                "given table name => '{}' has joined mergeable table => '{}' before".format(
                    small_table_name, mergeable_table.name
                )
            )

        hbm_vocab_size = config["device_vocabulary_size"]
        ddr_vocab_size = config["host_vocabulary_size"]
        ssd_vocab_size = config["ssd_vocabulary_size"]

        mergeable_table.merge_in(small_table_name, hbm_vocab_size, ddr_vocab_size, ssd_vocab_size)
        self._stable_to_mtable[small_table_name] = mergeable_table

        return mergeable_table

    def init_sliced_variables(self) -> None:
        for mtable in self._mtables:
            mtable.init_sliced_variable()

    def replace_mock_variables(self) -> None:
        for mtable in self._mtables:
            mtable.replace_mock_variable()
