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

from typing import Dict, Set, Any

from mx_rec.core.emb.dynamic_sparse_embedding import DynamicSparseEmbedding
from mx_rec.util.log import logger


class MergeableSparseEmbedding(DynamicSparseEmbedding):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._merged_small_tables: Set[str] = set()

    @property
    def merged_small_tables(self) -> Set[str]:
        return self._merged_small_tables

    def merge_in(self, small_table_name: str) -> None:
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
