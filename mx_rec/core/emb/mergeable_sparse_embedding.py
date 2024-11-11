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

import tensorflow as tf
from tensorflow import Tensor, Operation, Graph

from mx_rec.core.emb.sparse_embedding import SparseEmbedding
from mx_rec.util.initialize import ConfigInitializer
from mx_rec.util.log import logger
from mx_rec.validator.emb_validator import check_emb_init_params


class MergeableSparseEmbedding(SparseEmbedding):
    _mtable_id = 0

    _MERGEABLE_TABLE_PREFIX = "mergeable_table"
    _MOCK_VARIABLE_PREFIX = "mock_variable"

    def __init__(self, small_table_name: str, config: Dict[str, Any]) -> None:
        self._validate_small_tname(small_table_name)
        config["table_name"] = self._gen_mergeable_tname()

        super().__init__(config)

        self._mock_var_id: int = 0
        self._is_var_initialized: bool = False
        self._is_mock_var_replaced: bool = False
        self._deferred_lookup_funcs: List[Tuple[Tensor, Callable]] = []
        self._merged_small_tables: Set[str] = {small_table_name}

    @property
    def is_var_initialized(self) -> bool:
        return self._is_var_initialized

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
    def _gen_mergeable_tname(cls) -> str:
        mergeable_tname = cls._MERGEABLE_TABLE_PREFIX

        if cls._mtable_id != 0:
            mergeable_tname = "{}_{}".format(mergeable_tname, cls._mtable_id)

        cls._mtable_id += 1

        return mergeable_tname

    def merge_in(
        self, small_table_name: str, hbm_vocab_size: int = 0, ddr_vocab_size: int = 0, ssd_vocab_size: int = 0
    ) -> None:
        self._validate_small_tname(small_table_name)

        if small_table_name in self._merged_small_tables:
            raise ValueError(
                "given table => '{}' already exists in mergeable table => '{}'".format(
                    small_table_name, self.table_name
                )
            )

        # Recalculate sliced HBM, DDR and SSD vocabulary size.
        self._incr_hbm_vocab_size(hbm_vocab_size)
        self._incr_ddr_vocab_size(ddr_vocab_size)
        self._incr_ssd_vocab_size(ssd_vocab_size)
        self._set_slice_vocab_size()

        self._merged_small_tables.add(small_table_name)

        logger.info("Succeed to merge small table '%s' into large table '%s'.", small_table_name, self._table_name)

    def capacity(self) -> int:
        return self._device_vocabulary_size + self._host_vocabulary_size + self._ssd_vocabulary_size

    def create_mock_variable(self, feat_cnt: int) -> Tensor:
        if self.is_var_initialized:
            raise RuntimeError(
                "mergeable table => '{}' has allocated its variable, no mock variable should be generated"
            )

        mock_var_name = "{}_{}_{}".format(self._table_name, self._MOCK_VARIABLE_PREFIX, self._mock_var_id)
        self._mock_var_id += 1
        self._mock_var = tf.compat.v1.placeholder(
            tf.float32, shape=[None, feat_cnt, self._emb_size], name=mock_var_name
        )

        return self._mock_var

    def register_deferred_lookup(self, mock_variable: Tensor, deferred_lookup: Callable) -> None:
        pair = (mock_variable, deferred_lookup)
        self._deferred_lookup_funcs.append(pair)

    def replace_mock_variable(self) -> None:
        if self._is_mock_var_replaced:
            return

        g: Graph = tf.compat.v1.get_default_graph()
        ops: List[Operation] = g.get_operations()

        for mock_var, deferred_lookup in self._deferred_lookup_funcs:
            embs = deferred_lookup()
            consumers = [op for op in ops if mock_var in op.inputs]
            for consumer in consumers:
                idx = consumer.inputs.index(mock_var)
                consumer._update_input(index=idx, tensor=embs)

        self._is_mock_var_replaced = True
        logger.info("Mergeable table '%s' has replace its mock variable with lookup result.", self._table_name)

    def init_sliced_variable(self) -> None:
        if self._is_var_initialized:
            return

        super()._init_sliced_variable()

        self._is_var_initialized = True
        logger.info("Mergeable table '%s' has initialized its tf.Variable.", self._table_name)

    def _init_sliced_variable(self) -> None:
        logger.info("Mergeable table '%s' skipped initialization of tf.Variable.", self._table_name)

    def _incr_hbm_vocab_size(self, size: int) -> None:
        if size == 0:
            return

        origin = self._device_vocabulary_size
        self._device_vocabulary_size += size
        logger.info(
            "HBM vocabulary size of embedding table '%s' has been increased from '%s' to '%s'.",
            self.table_name,
            origin,
            self._device_vocabulary_size,
        )

    def _incr_ddr_vocab_size(self, size: int) -> None:
        if size == 0:
            return

        origin = self._host_vocabulary_size
        self._host_vocabulary_size += size
        logger.info(
            "DDR vocabulary size of embedding table '%s' has been increased from '%s' to '%s'.",
            self.table_name,
            origin,
            self._host_vocabulary_size,
        )

    def _incr_ssd_vocab_size(self, size: int) -> None:
        if size == 0:
            return

        origin = self._ssd_vocabulary_size
        self._ssd_vocabulary_size += size
        logger.info(
            "SSD vocabulary size of embedding table '%s' has been increased from '%s' to '%s'.",
            self.table_name,
            origin,
            self._ssd_vocabulary_size,
        )
