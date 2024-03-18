# coding=utf-8
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

import os
from glob import glob

import tensorflow as tf
from utils import FeatureSpecIns, create_feature_spec_list
from mx_rec.core.asc.helper import get_asc_insert_func


def input_fn(params, cfg, is_eval=False):
    def extract_fn(data_record):
        features = {
            # Extract features using the keys set during creation
            'label': tf.compat.v1.FixedLenFeature(shape=(cfg.line_per_sample,), dtype=tf.int64),
            'sparse_feature': tf.compat.v1.FixedLenFeature(shape=(26 * cfg.line_per_sample,), dtype=tf.int64),
            'dense_feature': tf.compat.v1.FixedLenFeature(shape=(13 * cfg.line_per_sample,), dtype=tf.float32),
        }
        sample = tf.compat.v1.parse_single_example(data_record, features)
        return sample

    def _reshape_fn(batch):
        batch['label'] = tf.reshape(batch['label'], [-1, 1])
        batch['dense_feature'] = tf.reshape(batch['dense_feature'], [-1, 13])
        batch['dense_feature'] = tf.math.log(batch['dense_feature'] + 3.0)
        batch['sparse_feature'] = tf.reshape(batch['sparse_feature'], [-1, 26])
        return batch

    if is_eval or params.run_mode == 'predict':
        files_list = glob(os.path.join(cfg.data_path, "test") + '/*.tfrecord')
    else:
        files_list = glob(os.path.join(cfg.data_path, "train") + '/*.tfrecord')

    num_parallel = 8
    dataset = tf.data.TFRecordDataset(files_list, num_parallel_reads=num_parallel)
    batch_size = cfg.batch_size // cfg.line_per_sample

    dataset = dataset.shard(cfg.rank_size, cfg.rank_id)
    dataset = dataset.shuffle(batch_size * 1000, seed=cfg.seed)
    dataset = dataset.repeat(cfg.train_epoch)
    dataset = dataset.map(extract_fn, num_parallel_calls=num_parallel).batch(batch_size,
                                                                             drop_remainder=True)
    dataset = dataset.map(_reshape_fn, num_parallel_calls=num_parallel)

    if not params.modify_graph:
        feature_spec_list = create_feature_spec_list(cfg)
        if is_eval:
            FeatureSpecIns.get_instance().set_eval_feature_spec_list(feature_spec_list)
            dataset = dataset.map(get_asc_insert_func(tgt_key_specs=feature_spec_list, is_training=False))
        else:
            FeatureSpecIns.get_instance().set_train_feature_spec_list(feature_spec_list)
            dataset = dataset.map(get_asc_insert_func(tgt_key_specs=feature_spec_list, is_training=True))
    dataset = dataset.prefetch(100)
    if params.use_one_shot:
        iterator = dataset.make_one_shot_iterator()
        batch = iterator.get_next()
        return batch
    return dataset
        # use one shot iterator
