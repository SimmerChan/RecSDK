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

import tensorflow as tf

from mx_rec.core.emb.sparse_embedding import SparseEmbedding
from mx_rec.util.initialize import ConfigInitializer
from mx_rec.util.log import logger


class MergeableSparseEmbedding(SparseEmbedding):
    _mtable_id = 0

    _MERGEABLE_TNAME_PREFIX = "mergeable_table"

    def __init__(self, table_name: str, config: Dict[str, Any]) -> None:
        self._validate_tname(table_name)
        config["table_name"] = self._gen_mergeable_tname()

        super().__init__(config)
        self._merged_table_names: Set[str] = set()
        self._merged_table_names.add(table_name)

    @classmethod
    def _validate_tname(cls, tname: str) -> None:
        if tname.startswith(cls._MERGEABLE_TNAME_PREFIX):
            raise ValueError(
                "original table name => '{}' is not supposed to start with '{}'".format(
                    tname, cls._MERGEABLE_TNAME_PREFIX
                )
            )

    @classmethod
    def _gen_mergeable_tname(cls) -> str:
        mergeable_tname = cls._MERGEABLE_TNAME_PREFIX

        if cls._mtable_id != 0:
            mergeable_tname = "{}_{}".format(mergeable_tname, cls._mtable_id)

        cls._mtable_id += 1

        return mergeable_tname

    @property
    def merged_table_names(self) -> Set[str]:
        return self._merged_table_names

    def merge_in(
        self, table_name: str, hbm_vocab_size: int = 0, ddr_vocab_size: int = 0, ssd_vocab_size: int = 0
    ) -> None:
        self._validate_tname(table_name)

        if table_name in self._merged_table_names:
            raise ValueError(
                "given table => '{}' already exists in mergeable table => '{}'".format(table_name, self.table_name)
            )

        self._incr_hbm_vocab_size(hbm_vocab_size)
        self._incr_ddr_vocab_size(ddr_vocab_size)
        self._incr_ssd_vocab_size(ssd_vocab_size)
        # Recalculate sliced HBM, DDR and SSD vocabulary size.
        self._set_slice_vocab_size()

        g_config = ConfigInitializer.get_instance()
        g_table_key = g_config.train_params_config.ascend_global_hashtable_collection
        g_table_vars = tf.compat.v1.get_collection_ref(g_table_key)

        old_var = self._variable
        old_var_shape = old_var.get_shape()

        new_var_name = "{}_{}".format(self._table_name, len(self.merged_table_names))
        new_var = tf.compat.v1.get_variable(
            name=new_var_name, shape=(self._slice_device_vocabulary_size, self._emb_size), trainable=False
        )
        new_var_shape = new_var.get_shape()

        g_table_vars.remove(old_var)
        g_table_vars.append(new_var)
        g_config.sparse_embed_config.update_table_instance(
            table_name=self._table_name, emb_table=self, old_var=old_var, new_var=new_var
        )

        self._variable = new_var
        self._merged_table_names.add(table_name)

        logger.info(
            "Succeed to merge small table '%s' into large table '%s', which has expanded its variable from %s -> %s.",
            table_name,
            self._table_name,
            old_var_shape,
            new_var_shape,
        )

    def capacity(self) -> int:
        return self._device_vocabulary_size + self._host_vocabulary_size + self._ssd_vocabulary_size

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
