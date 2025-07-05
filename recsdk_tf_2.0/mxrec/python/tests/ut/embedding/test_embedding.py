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

import pytest
import tensorflow as tf

import mxrec
from mxrec.python.embedding.table import StaticEmbTable
from mxrec.python.config import TomlParser
from mxrec.python.utils import logger
from mxrec.python.tests.ut.ut_utils import sess, test_graph


class TestGetEmbeddingTable:
    """Test for 'mxrec.python.embedding.embedding.get_embedding_table'."""

    @staticmethod
    def setup_method():
        mxrec.init("./ut_test.toml")

    @staticmethod
    def teardown_method():
        tf.compat.v1.reset_default_graph()
        TomlParser._instance = None
        logger._instance = None

    @staticmethod
    def test_get_static_table_ok():
        table = mxrec.get_embedding_table(
            name="test_name",
            dimension=8,
            device_vocabulary_size=100,
            initializer=tf.compat.v1.truncated_normal_initializer(),
            key_dtype=tf.int64,
            value_dtype=tf.float32,
        )
        assert isinstance(table, StaticEmbTable)

    @staticmethod
    def test_get_exist_table_ok():
        table1 = mxrec.get_embedding_table(
            name="test_name",
            dimension=8,
            device_vocabulary_size=100,
            initializer=tf.compat.v1.truncated_normal_initializer(),
            key_dtype=tf.int64,
            value_dtype=tf.float32,
        )
        table2 = mxrec.get_embedding_table(
            name="test_name",
            dimension=8,
            device_vocabulary_size=100,
            initializer=tf.compat.v1.truncated_normal_initializer(),
            key_dtype=tf.int64,
            value_dtype=tf.float32,
        )
        assert id(table1) == id(table2)

    @staticmethod
    def test_name_type_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.get_embedding_table(
                name=111,
                dimension=8,
                device_vocabulary_size=100,
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int64,
                value_dtype=tf.float32,
            )
        assert "is not str" in str(excinfo.value)

    @staticmethod
    def test_name_whitelist_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.get_embedding_table(
                name=r"test_name\n",
                dimension=8,
                device_vocabulary_size=100,
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int64,
                value_dtype=tf.float32,
            )
        assert "Note: It should be a string consisting of '[0-9A-Za-z_.-]'" in str(excinfo.value)

    @staticmethod
    def test_dev_voc_size_type_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.get_embedding_table(
                name="test_name",
                dimension=8,
                device_vocabulary_size="100",
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int64,
                value_dtype=tf.float32,
            )
        assert "is not int" in str(excinfo.value)

    @staticmethod
    def test_dev_voc_size_value_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.get_embedding_table(
                name="test_name",
                dimension=8,
                device_vocabulary_size=-1,
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int64,
                value_dtype=tf.float32,
            )
        assert "is less than" in str(excinfo.value)

    @staticmethod
    def test_dev_voc_size_only_dev_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.get_embedding_table(
                name="test_name",
                dimension=8,
                device_vocabulary_size=0,
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int64,
                value_dtype=tf.float32,
            )
        assert "currently, the embedding table only supports storage in device memory" in str(excinfo.value)

    @staticmethod
    def test_initializer_type_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.get_embedding_table(
                name="test_name",
                dimension=8,
                device_vocabulary_size=100,
                initializer="tf.compat.v1.truncated_normal_initializer()",
                key_dtype=tf.int64,
                value_dtype=tf.float32,
            )
        assert "type of 'initializer'" in str(excinfo.value)

    @staticmethod
    def test_key_dtype_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.get_embedding_table(
                name="test_name",
                dimension=8,
                device_vocabulary_size=100,
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int32,
                value_dtype=tf.float32,
            )
        assert "only supports key dtype of tf.int64" in str(excinfo.value)

    @staticmethod
    def test_value_dtype_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.get_embedding_table(
                name="test_name",
                dimension=8,
                device_vocabulary_size=100,
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int64,
                value_dtype=tf.float16,
            )
        assert "only supports value dtype of tf.float32" in str(excinfo.value)

    @staticmethod
    def test_dist_strategy_type_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.get_embedding_table(
                name="test_name",
                dimension=8,
                device_vocabulary_size=100,
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int64,
                value_dtype=tf.float32,
                distribution_strategy=1,
            )
        assert "is not str" in str(excinfo.value)

    @staticmethod
    def test_dist_strategy_whitelist_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.get_embedding_table(
                name="test_name",
                dimension=8,
                device_vocabulary_size=100,
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int64,
                value_dtype=tf.float32,
                distribution_strategy=r"MP\n",
            )
        assert "Note: It should be a string consisting of '[0-9A-Za-z_.-]'" in str(excinfo.value)

    @staticmethod
    def test_dist_strategy_invalid_value_err():
        with pytest.raises(ValueError) as excinfo:
            mxrec.get_embedding_table(
                name="test_name",
                dimension=8,
                device_vocabulary_size=100,
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int64,
                value_dtype=tf.float32,
                distribution_strategy="UNKNOWN",
            )
        assert "the embedding table supports MP(model parallelism) and DP(data parallelism)" in str(excinfo.value)

    @staticmethod
    def test_get_init_hashtable_op_ok():
        mxrec.get_embedding_table(
            name="test_name",
            dimension=8,
            device_vocabulary_size=100,
            initializer=tf.compat.v1.truncated_normal_initializer(),
            key_dtype=tf.int64,
            value_dtype=tf.float32,
        )
        init_ops = mxrec.get_init_hashtable_op()
        assert len(init_ops) != 0

    @staticmethod
    def test_min_used_times_type_err():
        with pytest.raises(ValueError):
            mxrec.get_embedding_table(
                name="test_name",
                dimension=8,
                device_vocabulary_size=100,
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int64,
                value_dtype=tf.float32,
                min_used_times="invalid",
            )

    @staticmethod
    def test_min_used_times_upper_bound_err():
        with pytest.raises(ValueError):
            mxrec.get_embedding_table(
                name="test_name",
                dimension=8,
                device_vocabulary_size=100,
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int64,
                value_dtype=tf.float32,
                min_used_times=-1,
            )

    @staticmethod
    def test_min_used_times_lower_bound_err():
        with pytest.raises(ValueError):
            mxrec.get_embedding_table(
                name="test_name",
                dimension=8,
                device_vocabulary_size=100,
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int64,
                value_dtype=tf.float32,
                min_used_times=1 << 31,
            )

    @staticmethod
    def test_max_cold_secs_type_err():
        with pytest.raises(ValueError):
            mxrec.get_embedding_table(
                name="test_name",
                dimension=8,
                device_vocabulary_size=100,
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int64,
                value_dtype=tf.float32,
                max_cold_secs="invalid",
            )

    @staticmethod
    def test_max_cold_secs_upper_bound_err():
        with pytest.raises(ValueError):
            mxrec.get_embedding_table(
                name="test_name",
                dimension=8,
                device_vocabulary_size=100,
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int64,
                value_dtype=tf.float32,
                max_cold_secs=-1,
            )

    @staticmethod
    def test_max_cold_secs_lower_bound_err():
        with pytest.raises(ValueError):
            mxrec.get_embedding_table(
                name="test_name",
                dimension=8,
                device_vocabulary_size=100,
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int64,
                value_dtype=tf.float32,
                max_cold_secs=1 << 64,
            )


class TestEmbeddingLookup:
    """Test for 'mxrec.python.embedding.embedding.embedding_lookup'."""

    @staticmethod
    def setup_method():
        tf.compat.v1.disable_eager_execution()
        mxrec.init("./ut_test.toml")

    @staticmethod
    def teardown_method():
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.enable_eager_execution()
        TomlParser._instance = None
        logger._instance = None

    @staticmethod
    def test_mp_lookup_ok():
        with test_graph.as_default():
            table = mxrec.get_embedding_table(
                name="test_name",
                dimension=8,
                device_vocabulary_size=100,
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int64,
                value_dtype=tf.float32,
            )
            ids = tf.constant([[1, 2, 4, 1], [1, 3, 10, 2]], dtype=tf.int64)
            lookup_res = mxrec.embedding_lookup(table, ids)

            assert len(mxrec.get_sparse_embedding()) > 0

            sess.run(tf.compat.v1.global_variables_initializer())
            res = sess.run(lookup_res)
            # The shape must be [bs, seq_len, emb_dim].
            assert res.shape == (2, 4, 8)

    @staticmethod
    def test_params_type_err():
        ids = tf.constant([[1, 2, 4, 1], [1, 3, 10, 2]], dtype=tf.int64)
        with pytest.raises(ValueError) as excinfo:
            mxrec.embedding_lookup("XXX", ids)
        assert "type of 'emb_table'" in str(excinfo.value)

    @staticmethod
    def test_ids_type_err():
        table = mxrec.get_embedding_table(
            name="test_name",
            dimension=8,
            device_vocabulary_size=100,
            initializer=tf.compat.v1.truncated_normal_initializer(),
            key_dtype=tf.int64,
            value_dtype=tf.float32,
        )
        with pytest.raises(ValueError) as excinfo:
            mxrec.embedding_lookup(table, "1")
        assert "type of 'ids'" in str(excinfo.value)
