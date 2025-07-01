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

from typing import Tuple

import tensorflow as tf


class LearningRateScheduler:
    """LR Scheduler combining Polynomial Decay with Warmup at the beginning."""

    def __init__(
        self, base_lr_dense: float, base_lr_sparse: float, warmup_steps: int, decay_start_step: int, decay_steps: int
    ):
        self._warmup_steps = tf.constant(warmup_steps, dtype=tf.int32)
        self._decay_start_step = tf.constant(decay_start_step, dtype=tf.int32)
        self._decay_steps = tf.constant(decay_steps)
        self._decay_end_step = decay_start_step + decay_steps
        self._poly_power = 2.0
        self._base_lr_dense = base_lr_dense
        self._base_lr_sparse = base_lr_sparse

    def calc(self, global_step: tf.Variable) -> Tuple[tf.Tensor, tf.Tensor]:
        # Used for the warmup stage.
        warmup_step = tf.cast(1 / self._warmup_steps, tf.float32)
        lr_factor_warmup = 1 - tf.cast(self._warmup_steps - global_step, tf.float32) * warmup_step
        lr_factor_warmup = tf.cast(lr_factor_warmup, tf.float32)
        # Used for the constant stage.
        lr_factor_constant = tf.cast(1.0, tf.float32)

        # Used for the decay stage.
        lr_factor_decay = (self._decay_end_step - global_step) / self._decay_steps
        lr_factor_decay = tf.math.pow(lr_factor_decay, self._poly_power)
        lr_factor_decay = tf.cast(lr_factor_decay, tf.float32)
        sparse_after_decay = tf.cast(1 / self._decay_steps, tf.float32)

        lr_factor_decay_sparse = tf.cond(
            global_step < self._decay_end_step,
            lambda: lr_factor_decay,
            lambda: sparse_after_decay,
        )

        lr_factor_decay_dense = tf.cond(
            global_step < self._decay_end_step,
            lambda: lr_factor_decay,
            lambda: sparse_after_decay,
        )

        poly_schedule_sparse = tf.cond(
            global_step < self._decay_start_step,
            lambda: lr_factor_constant,
            lambda: lr_factor_decay_sparse,
        )

        poly_schedule_dense = tf.cond(
            global_step < self._decay_start_step,
            lambda: lr_factor_constant,
            lambda: lr_factor_decay_dense,
        )

        lr_factor_sparse = tf.cond(
            global_step < self._warmup_steps, lambda: lr_factor_warmup, lambda: poly_schedule_sparse
        )

        lr_factor_dense = tf.cond(
            global_step < self._warmup_steps, lambda: lr_factor_warmup, lambda: poly_schedule_dense
        )

        lr_sparse = self._base_lr_sparse * lr_factor_sparse
        lr_dense = self._base_lr_dense * lr_factor_dense
        return lr_dense, lr_sparse
