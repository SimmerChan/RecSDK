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

from typing import Tuple

import tensorflow as tf

from mxrec.python.constants import StaticEmbTableConfig
from mxrec.python.utils import gen_npu_cpu_ops
from mxrec.python.embedding.table.base_emb_table import BaseEmbTable
from mxrec.python.embedding.feature import CountFilter, TimeEvictor


class StaticEmbTable(BaseEmbTable):
    """An embedding table with a fixed table size. Currently, the table is stored in device memory."""

    def __init__(self, et_config: StaticEmbTableConfig):
        super(StaticEmbTable, self).__init__(et_config)

        self._count_filter = CountFilter(et_config.name, et_config.min_used_times) if et_config.min_used_times else None
        self._time_evictor = TimeEvictor(et_config.name, et_config.max_cold_secs) if et_config.max_cold_secs else None

    @property
    def count_filter(self) -> CountFilter:
        return self._count_filter

    @property
    def time_evictor(self) -> TimeEvictor:
        return self._time_evictor

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def export(self) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError

    def assign(self, keys: tf.Tensor, embeddings: tf.Tensor):
        raise NotImplementedError

    def _create_hashtable(self) -> tf.Tensor:
        table_handle = gen_npu_cpu_ops.init_embedding_hashmap_v2(
            table_id=self._table_id,
            bucket_size=self.slice_dev_vocab_size,
            embedding_dim=self.dim,
            load_factor=self._load_factor,
            dtype=self.value_dtype,
        )
        sampled_values = self._et_config.initializer(shape=[self.slice_dev_vocab_size, self.dim], dtype=tf.float32)
        # The `initializer_mode` supports two modes: "random" and "constant". When the parameter is set to "constant",
        # it must be used together with `constant_value`. When the parameter is set to "random", it must be used
        # together with `sampled_values`.
        hashtable = gen_npu_cpu_ops.init_embedding_hash_table(
            table_handle=table_handle,
            bucket_size=self.slice_dev_vocab_size,
            embedding_dim=self.dim,
            initializer_mode="random",
            sampled_values=sampled_values,
        )
        return hashtable
