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
from mxrec.python.embedding.feature.evictor import TimeEvictor
from mxrec.python.utils import gen_npu_cpu_ops


@pytest.fixture
def mock_get_device_id(monkeypatch: pytest.MonkeyPatch):
    def mock_get_device_id():
        return 0

    monkeypatch.setattr("mxrec.python.binding.runtime_manager.get_device_id", mock_get_device_id)


class TestTimeEvictor:
    @staticmethod
    def teardown_method():
        tf.compat.v1.reset_default_graph()

    @staticmethod
    def test_run_update_op_twice(mock_get_device_id):
        time_evictor = TimeEvictor(table_name="test_table_2", max_cold_secs=2)

        with test_graph.as_default():
            data = {
                "keys": tf.constant([[1, 2, 3, 4, 5]], dtype=tf.int64),
            }
            dataset = tf.data.Dataset.from_tensor_slices(data)
            dataset = dataset.repeat(count=2)

            iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
            batch = iterator.get_next()

            keys = time_evictor.update_last_timestamp(batch["keys"])

            res: np.ndarray = None
            for _ in range(2):
                res = sess.run(keys)

        assert res.tolist() == [1, 2, 3, 4, 5]

    @staticmethod
    def test_evict_op_ok():
        init_op = None
        evict_op = None
        embedding = None

        table_id = 0
        table_cap = 100
        num_dim = 8
        dtype = tf.float32

        with test_graph.as_default():
            table_handle = gen_npu_cpu_ops.init_embedding_hashmap_v2(
                table_id=table_id, bucket_size=table_cap, embedding_dim=num_dim, load_factor=0.8, dtype=dtype
            )
            init_op = gen_npu_cpu_ops.init_embedding_hash_table(
                table_handle=table_handle,
                bucket_size=table_cap,
                embedding_dim=num_dim,
                initializer_mode="random",
                sampled_values=tf.compat.v1.random_normal_initializer()(shape=[table_cap, num_dim], dtype=dtype),
            )

            table_handle = gen_npu_cpu_ops.table_to_resource_v2(table_id=[table_id])

            lookup_ids = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
            embedding = gen_npu_cpu_ops.embedding_hash_table_lookup_or_insert(
                table_handle=table_handle, keys=lookup_ids, bucket_size=table_cap, embedding_dim=num_dim
            )

            evict_ids = tf.constant([1, 3, 5], dtype=tf.int64)
            evict_op = gen_npu_cpu_ops.embedding_hash_table_evict(
                table_handle=table_handle, keys=evict_ids, table_cap=table_cap, embedding_dim=num_dim
            )

        with sess.as_default():
            sess.run(init_op)

            embedding = sess.run(embedding)
            assert np.array_equal(embedding, np.zeros(shape=(lookup_ids.shape[0], num_dim)))

            sess.run(evict_op)
