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

import numpy as np
import pytest
import tensorflow as tf

from mxrec.python.tests.ut.ut_utils import sess, test_graph
from mxrec.python.embedding.feature.filter import CountFilter


@pytest.fixture
def mock_get_device_id(monkeypatch: pytest.MonkeyPatch):
    def mock_get_device_id():
        return 0

    monkeypatch.setattr("mxrec.python.binding.runtime_manager.get_device_id", mock_get_device_id)


class TestCountFilter:
    @staticmethod
    def teardown_method():
        tf.compat.v1.reset_default_graph()

    @staticmethod
    def test_run_filter_op_twice(mock_get_device_id):
        count_filter = CountFilter(table_name="test_table_2", min_used_times=2)

        with test_graph.as_default():
            data = {
                "keys": tf.constant([[1, 2, 3, 4, 5]], dtype=tf.int64),
                "cnts": tf.constant([[1, 1, 0, 0, 0]], dtype=tf.int32),
            }
            dataset = tf.data.Dataset.from_tensor_slices(data)
            dataset = dataset.repeat(2)

            iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
            batch = iterator.get_next()

            filtered_keys = count_filter.count_and_filter(batch["keys"], batch["cnts"])

            res: np.ndarray = None
            res = sess.run(filtered_keys)
            # The current simulation environment has not yet completed the adaptation of D2H/H2D.
            assert res == 0
