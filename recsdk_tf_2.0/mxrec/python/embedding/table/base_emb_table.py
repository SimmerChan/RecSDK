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

import abc
from dataclasses import fields
from typing import Tuple, Dict

import tensorflow as tf

from mxrec.python.communication import get_rank_size
from mxrec.python.constants import EmbTableConfig, EmbDistributionStrategy, INIT_HASHTABLE_COLLECTION


class BaseEmbTableInterface(metaclass=abc.ABCMeta):
    """Base class interface of embedding table."""

    @abc.abstractmethod
    def save(self):
        raise NotImplementedError

    @abc.abstractmethod
    def load(self):
        raise NotImplementedError

    def export(self) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError

    def assign(self, keys: tf.Tensor, embeddings: tf.Tensor) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _create_hashtable(self) -> tf.Tensor:
        raise NotImplementedError


class BaseEmbTable(BaseEmbTableInterface):
    """Base class of embedding table."""

    _global_table_id = 0
    _table_ins_to_id = {}

    def __init__(self, et_config: EmbTableConfig):
        self._et_config = et_config
        self._table_id = BaseEmbTable._global_table_id
        # The load factor is used for the dynamic table and is currently not effective.
        self._load_factor = 0.8
        self._hashtable = None
        self._slice_dev_vocab_size = None
        self._emb_table_init()

        # Record and update the embedding info.
        BaseEmbTable._table_ins_to_id[self] = BaseEmbTable._global_table_id
        BaseEmbTable._global_table_id += 1

    def __repr__(self) -> str:
        attrs = []
        for field in fields(self._et_config):
            field_value = getattr(self._et_config, field.name)
            attr = f"{field.name}='{field_value}'" if isinstance(field_value, str) else f"{field.name}={field_value}"
            attrs.append(attr)
        attrs.append(f"id={self._table_id}")
        fmt_attrs = ", ".join(attrs)
        fmt_repr = f"{self.__class__.__name__}({fmt_attrs})"

        return fmt_repr

    @property
    def name(self):
        return self._et_config.name

    @property
    def dim(self):
        return self._et_config.dim

    @property
    def dev_vocab_size(self):
        return self._et_config.dev_vocab_size

    @property
    def initializer(self):
        return self._et_config.initializer

    @property
    def value_dtype(self):
        return self._et_config.value_dtype

    @property
    def key_dtype(self):
        return self._et_config.key_dtype

    @property
    def dist_strategy(self):
        return self._et_config.dist_strategy

    @property
    def slice_dev_vocab_size(self):
        return self._slice_dev_vocab_size

    @property
    def table_id(self):
        return self._table_id

    @classmethod
    def get_table_ins_to_id(cls) -> Dict["BaseEmbTable", int]:
        return cls._table_ins_to_id

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def export(self) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError

    def assign(self, keys: tf.Tensor, embeddings: tf.Tensor) -> None:
        raise NotImplementedError

    def _create_hashtable(self) -> tf.Tensor:
        raise NotImplementedError

    def _emb_table_init(self):
        if self.dist_strategy == EmbDistributionStrategy.DP.value:
            self._slice_dev_vocab_size = self.dev_vocab_size
        else:
            self._slice_dev_vocab_size = int(self.dev_vocab_size / get_rank_size())
        self._hashtable = self._create_hashtable()
        tf.compat.v1.add_to_collection(INIT_HASHTABLE_COLLECTION, self._hashtable)
