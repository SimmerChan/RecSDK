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

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

import mxrec
from mxrec.python.optimizer.utils import deduplicate_indexed_slices
from mxrec.python.config import TomlParser
from mxrec.python.utils import logger


class TestDeduplicateIndexedSlices:
    """Test for 'mxrec.python.optimizer.utils.deduplicate_indexed_slices'."""

    @staticmethod
    def setup_method():
        mxrec.init("./ut_test.toml")
        tf.compat.v1.disable_eager_execution()

    @staticmethod
    def teardown_method():
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.enable_eager_execution()
        TomlParser._instance = None
        logger._instance = None

    @staticmethod
    def test_deduplicate_ok():
        grad = ops.IndexedSlices(
            values=tf.constant([0.1, 0.4, 0.1], tf.float32),
            indices=tf.constant([0, 1, 0], tf.int64),
            dense_shape=tf.convert_to_tensor((4, 4)),
        )

        deduplicate_grad = deduplicate_indexed_slices(grad)

        with tf.compat.v1.Session() as sess:
            deduplicate_values = sess.run(deduplicate_grad.values)
        assert np.array_equal(deduplicate_values, np.array([0.2, 0.4], np.float32))
