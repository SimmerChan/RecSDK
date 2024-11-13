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

from typing import Dict, Set, Any, Callable, List, Tuple

from tensorflow import Tensor

from mx_rec.core.emb.dynamic_sparse_embedding import DynamicSparseEmbedding
from mx_rec.util.log import logger


class MergeableSparseEmbedding(DynamicSparseEmbedding):
    _mtable_id = 0

    _MERGEABLE_TABLE_PREFIX = "mergeable_table"

    def __init__(self, small_table_name: str, config: Dict[str, Any]) -> None:
        self._validate_small_tname(small_table_name)
        config["table_name"] = self._gen_mergeable_table_name()

        super().__init__(config)

        self._mock_var_id: int = 0
        self._deferred_lookup_funcs: List[Tuple[Tensor, Callable]] = []
        self._merged_small_tables: Set[str] = {small_table_name}

    @property
    def merged_small_tables(self) -> Set[str]:
        return self._merged_small_tables

    @classmethod
    def _validate_small_tname(cls, tname: str) -> None:
        if tname.startswith(cls._MERGEABLE_TABLE_PREFIX):
            raise ValueError(
                "original table name => '{}' is not supposed to start with '{}'".format(
                    tname, cls._MERGEABLE_TABLE_PREFIX
                )
            )

    @classmethod
    def _gen_mergeable_table_name(cls) -> str:
        mergeable_tname = cls._MERGEABLE_TABLE_PREFIX

        if cls._mtable_id != 0:
            mergeable_tname = "{}_{}".format(mergeable_tname, cls._mtable_id)

        cls._mtable_id += 1

        return mergeable_tname

    def merge_in(self, small_table_name: str) -> None:
        self._validate_small_tname(small_table_name)

        if small_table_name in self._merged_small_tables:
            raise ValueError(
                "given table => '{}' already exists in mergeable table => '{}'".format(
                    small_table_name, self.table_name
                )
            )

        self._merged_small_tables.add(small_table_name)
        logger.info("Succeed to merge small table '%s' into large table '%s'.", small_table_name, self._table_name)

    def _set_slice_vocab_size(self):
        """Device vocabulary size will be forced set to 1 in dynamic expansion mode."""
        self._slice_device_vocabulary_size = 1
