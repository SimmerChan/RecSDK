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
from typing import Any, Dict, Optional, List

import tensorflow as tf
from tensorflow.python.ops.init_ops import Initializer as InitializerV1
from tensorflow.python.ops.init_ops_v2 import Initializer as InitializerV2

from mx_rec.core.emb.emb_factory import MergeableSparseEmbeddingFactory
from mx_rec.core.emb.sparse_embedding import SparseEmbedding
from mx_rec.core.emb.mergeable_sparse_embedding import MergeableSparseEmbedding
from mx_rec.util.singleton import singleton


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
        self._tname_to_mtable: Dict[str, SparseEmbedding] = {}

    def create_mergeable_table(self, union_key: UnionKey, table_name: str, config: Dict[str, Any]) -> SparseEmbedding:
        union_key.validate()

        mergeable_table = MergeableSparseEmbeddingFactory().create_embedding(table_name, config=config)
        self._mtables.append(mergeable_table)

        self._ukey_to_mtable[union_key] = mergeable_table
        self._tname_to_mtable[table_name] = mergeable_table

        return mergeable_table

    def find_mergeable_table(self, union_key: Optional[UnionKey], table_name: str = "") -> Optional[SparseEmbedding]:
        union_key.validate()

        if union_key and table_name:
            raise ValueError(
                "at most one of union key and table name should be provided, got union key => '{}' and table name => '{}'".format(
                    union_key, table_name
                )
            )

        if not (union_key or table_name):
            raise ValueError(
                "at least one of union key and table name should be provided, got union key => '{}' and table name => '{}'".format(
                    union_key, table_name
                )
            )

        if not union_key:
            return self._tname_to_mtable.get(table_name, None)

        return self._ukey_to_mtable.get(union_key, None)

    def join_mergeable_table(
        self, mergeable_table: MergeableSparseEmbedding, table_name: str, config: Dict[str, Any]
    ) -> MergeableSparseEmbedding:
        if table_name in mergeable_table.merged_table_names:
            raise ValueError(
                "given table name => '{}' has joined mergeable table => '{}' before".format(
                    table_name, mergeable_table.name
                )
            )

        hbm_vocab_size = config["device_vocabulary_size"]
        ddr_vocab_size = config["host_vocabulary_size"]
        ssd_vocab_size = config["ssd_vocabulary_size"]

        mergeable_table.merge_in(table_name, hbm_vocab_size, ddr_vocab_size, ssd_vocab_size)
        self._tname_to_mtable[table_name] = mergeable_table

        return mergeable_table
