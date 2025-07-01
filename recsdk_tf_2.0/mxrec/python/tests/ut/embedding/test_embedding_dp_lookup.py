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

import tensorflow as tf

import mxrec
from mxrec.python.config import TomlParser
from mxrec.python.utils import logger
from mxrec.python.tests.ut.ut_utils import sess, test_graph


class TestDPLookup:
    """Test for 'mxrec.python.embedding.embedding.embedding_lookup.dp_lookup'."""

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
    def test_dp_lookup_ok():
        with test_graph.as_default():
            table = mxrec.get_embedding_table(
                name="test_name",
                dimension=8,
                device_vocabulary_size=100,
                initializer=tf.compat.v1.truncated_normal_initializer(),
                key_dtype=tf.int64,
                value_dtype=tf.float32,
                distribution_strategy="DP",
            )
            ids = tf.constant([[1, 2, 4, 1], [1, 3, 10, 2]], dtype=tf.int64)
            lookup_res = mxrec.embedding_lookup(table, ids)

            sess.run(tf.compat.v1.global_variables_initializer())
            res = sess.run(lookup_res)
            # The shape must be [bs, seq_len, emb_dim].
            assert res.shape == (2, 4, 8)
