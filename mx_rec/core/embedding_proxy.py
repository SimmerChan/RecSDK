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
from typing import Any, Dict, List, Optional, Union

import tensorflow as tf
from tensorflow.python.ops.init_ops import Initializer as InitializerV1
from tensorflow.python.ops.init_ops_v2 import Initializer as InitializerV2

from mx_rec.core.emb.mergeable_sparse_embedding import MergeableSparseEmbedding
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
    _MERGEABLE_TABLE_PREFIX = "mergeable_table"

    def __init__(self) -> None:
        self._mtable_id: int = 0
        self._mtables: List[MergeableSparseEmbedding] = []
        self._ukey_to_mtable: Dict[UnionKey, MergeableSparseEmbedding] = {}
        self._stable_to_mtable: Dict[str, MergeableSparseEmbedding] = {}

    def reset(self) -> None:
        self.__init__()

    @classmethod
    def _validate_small_tname(cls, tname: str) -> None:
        if tname.startswith(cls._MERGEABLE_TABLE_PREFIX):
            raise ValueError(
                "original table name => '{}' is not supposed to start with '{}'".format(
                    tname, cls._MERGEABLE_TABLE_PREFIX
                )
            )

    def create_mergeable_table(
        self, union_key: UnionKey, small_table_name: str, config: Dict[str, Any]
    ) -> MergeableSparseEmbedding:
        union_key.validate()
        self._validate_small_tname(small_table_name)

        config["table_name"] = self._gen_mergeable_table_name()
        mergeable_table = MergeableSparseEmbedding(config)
        mergeable_table.merge_in(small_table_name)

        self._mtables.append(mergeable_table)
        self._ukey_to_mtable[union_key] = mergeable_table
        self._stable_to_mtable[small_table_name] = mergeable_table

        logger.info(
            "A new mergeable embedding table '%s' has been created from embedding table '%s'.",
            mergeable_table.table_name,
            small_table_name,
        )

        return mergeable_table

    def find_mergeable_table(self, key: Union[UnionKey, str]) -> Optional[MergeableSparseEmbedding]:
        if isinstance(key, UnionKey):
            key.validate()
            return self._ukey_to_mtable.get(key)
        elif isinstance(key, str):
            self._validate_small_tname(key)
            return self._stable_to_mtable.get(key)
        else:
            invalid_type = type(key)
            ukey_type = type(UnionKey)
            raise TypeError("not supported key type => '{}', expected '{}' or 'str'".format(invalid_type, ukey_type))

    def join_mergeable_table(
        self, mergeable_table: MergeableSparseEmbedding, small_table_name: str
    ) -> MergeableSparseEmbedding:
        if small_table_name in mergeable_table.merged_small_tables:
            raise ValueError(
                "given table name => '{}' has joined mergeable table => '{}' already".format(
                    small_table_name, mergeable_table.table_name
                )
            )

        mergeable_table.merge_in(small_table_name)
        self._stable_to_mtable[small_table_name] = mergeable_table

        return mergeable_table

    def _gen_mergeable_table_name(self) -> str:
        mergeable_tname = "{}_{}".format(self._MERGEABLE_TABLE_PREFIX, self._mtable_id)
        self._mtable_id += 1

        return mergeable_tname
