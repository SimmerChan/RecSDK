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

import tensorflow as tf


class LearningRateScheduler:
    """
    LR Scheduler combining Polynomial Decay with Warmup at the beginning.
    TF-based cond operations necessary for performance in graph mode.
    """

    def __init__(self, base_lr_dense, base_lr_sparse, warmup_steps, decay_start_step, decay_steps):
        self.warmup_steps = tf.constant(warmup_steps, dtype=tf.int32)
        self.decay_start_step = tf.constant(decay_start_step, dtype=tf.int32)
        self.decay_steps = tf.constant(decay_steps)
        self.decay_end_step = decay_start_step + decay_steps  # 65041
        self.poly_power = 2.0
        self.base_lr_dense = base_lr_dense
        self.base_lr_sparse = base_lr_sparse

    def calc(self, global_step):
        # used for the warmup stage
        warmup_step = tf.cast(1 / self.warmup_steps, tf.float32)
        lr_factor_warmup = 1 - tf.cast(self.warmup_steps - global_step, tf.float32) * warmup_step
        lr_factor_warmup = tf.cast(lr_factor_warmup, tf.float32)
        # used for the constant stage
        lr_factor_constant = tf.cast(1.0, tf.float32)

        # used for the decay stage
        lr_factor_decay = (self.decay_end_step - global_step) / self.decay_steps
        lr_factor_decay = tf.math.pow(lr_factor_decay, self.poly_power)
        lr_factor_decay = tf.cast(lr_factor_decay, tf.float32)
        sparse_after_decay = tf.cast(1 / self.decay_steps, tf.float32)

        lr_factor_decay_sparse = tf.cond(
            global_step < self.decay_end_step,
            lambda: lr_factor_decay,
            lambda: sparse_after_decay,
        )

        lr_factor_decay_dense = tf.cond(
            global_step < self.decay_end_step,
            lambda: lr_factor_decay,
            lambda: sparse_after_decay,
        )

        poly_schedule_sparse = tf.cond(
            global_step < self.decay_start_step,
            lambda: lr_factor_constant,
            lambda: lr_factor_decay_sparse,
        )

        poly_schedule_dense = tf.cond(
            global_step < self.decay_start_step,
            lambda: lr_factor_constant,
            lambda: lr_factor_decay_dense,
        )

        lr_factor_sparse = tf.cond(
            global_step < self.warmup_steps, lambda: lr_factor_warmup, lambda: poly_schedule_sparse
        )

        lr_factor_dense = tf.cond(
            global_step < self.warmup_steps, lambda: lr_factor_warmup, lambda: poly_schedule_dense
        )

        lr_sparse = self.base_lr_sparse * lr_factor_sparse
        lr_dense = self.base_lr_dense * lr_factor_dense
        return lr_dense, lr_sparse


class Config:
    def __init__(self, args):
        self.rank_id = int(os.getenv("RANK_ID")) if os.getenv("RANK_ID") else None
        self.rank_size = args.rank_size

        self.data_path = args.data_path
        self.train_file_pattern = "train"
        self.test_file_pattern = "test"

        self.batch_size = 8192
        self.line_per_sample = 1024
        self.train_epoch = 3
        self.test_epoch = 1
        self.perform_shuffle = False
        self.run_mode = args.run_mode
        self.key_type = tf.int64
        self.label_type = tf.float32
        self.value_type = tf.int64
        self.feat_cnt = 26

        self.seed = 128
        self.field_num = 26
        self.send_count = 46000 // self.rank_size
        self.emb_dim = 128
        self.hashtable_threshold = 1
        self.cache_mode = args.cache_mode
        self.use_dynamic = args.use_dynamic
        self.use_dynamic_expansion = args.use_dynamic_expansion
        self.modify_graph = args.modify_graph
        self.__set_emb_table_size()
        # 动态学习率
        global_batch_size = self.batch_size * args.rank_size
        lr_scheduler_steps = [
            int(2750 * 55296 / global_batch_size),
            int(49315 * 55296 / global_batch_size),
            int(27772 * 55296 / global_batch_size),
        ]
        self.global_step = tf.Variable(0, trainable=False)
        _lr_scheduler = LearningRateScheduler(
            28.443,
            33.71193,
            lr_scheduler_steps[0],
            lr_scheduler_steps[1],
            lr_scheduler_steps[2],
        )
        self.learning_rate = _lr_scheduler.calc(self.global_step)

    def get_emb_table_cfg(self) -> dict:
        if self.cache_mode == "HBM":
            return {"device_vocabulary_size": self.dev_vocab_size}
        if self.cache_mode == "DDR":
            return {"device_vocabulary_size": self.dev_vocab_size,
                    "host_vocabulary_size": self.host_vocab_size}
        if self.cache_mode == "SSD":
            return {"device_vocabulary_size": self.dev_vocab_size,
                    "host_vocabulary_size": self.host_vocab_size,
                    "ssd_vocabulary_size": self.ssd_vocab_size,
                    "ssd_data_path": ["ssd_data"]}
        raise RuntimeError(f"get CACHE_MODE:{self.cache_mode}, check Config.__set_emb_table_size implementation")

    def __set_emb_table_size(self):
        if self.cache_mode is None:
            raise ValueError("please export CACHE_MODE environment variable, support:[HBM, DDR, SSD]")
        if self.cache_mode == "HBM":
            self.dev_vocab_size = 23_000_000 * self.rank_size
            self.host_vocab_size = 0
            return
        if self.cache_mode == "DDR":
            if not self.use_dynamic:
                raise ValueError("for now, DDR mode only support dynamic shape, please export USE_DYNAMIC=1")
            self.dev_vocab_size = 500_000 * self.rank_size
            self.host_vocab_size = 24_000_000 * self.rank_size
            return
        if self.cache_mode == "SSD":
            if not self.use_dynamic:
                raise ValueError("for now, SSD mode only support dynamic shape, please export USE_DYNAMIC=1")
            self.dev_vocab_size = 100_000 * self.rank_size
            self.host_vocab_size = 2_000_000 * self.rank_size
            self.ssd_vocab_size = 24_000_000 * self.rank_size
            return
        raise ValueError(f"get CACHE_MODE:{self.cache_mode}, expect in [HBM, DDR, SSD]")

