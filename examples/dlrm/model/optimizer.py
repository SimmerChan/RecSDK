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

import tensorflow as tf
from delay_loss_scale import DenseLossScaleOptimizer, SparseLossScaleOptimizer

from mx_rec.util.initialize import ConfigInitializer
from mx_rec.optimizers.gradient_descent_by_addr import create_hash_optimizer_by_addr


def get_dense_and_sparse_optimizer(cfg):
    use_dynamic_expansion = ConfigInitializer.get_instance().use_dynamic_expansion
    if cfg.use_lazy_adam_optimizer:
        if use_dynamic_expansion:
            raise RuntimeError("model is incompatible with dynamic_expansion when use lazy_adam optimizer.")
        # use lazy_adam optimizer
        dense_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=cfg.learning_rate[0])
        from mx_rec.optimizers.lazy_adam import create_hash_optimizer
        sparse_optimizer = create_hash_optimizer(learning_rate=cfg.learning_rate[1])
    else:
        # use SGD optimizer
        dense_optimizer = tf.train.GradientDescentOptimizer(learning_rate=cfg.learning_rate[0])
        if use_dynamic_expansion:
            sparse_optimizer = create_hash_optimizer_by_addr(learning_rate=cfg.learning_rate[1], weight_decay=0.0001)
        else:
            from gradient_descent_w import create_hash_optimizer
            sparse_optimizer = create_hash_optimizer(learning_rate=cfg.learning_rate[1], weight_decay=0.0001)

    sparse_optimizer = SparseLossScaleOptimizer(sparse_optimizer, 1024, cfg)
    dense_optimizer = DenseLossScaleOptimizer(dense_optimizer, 1024, cfg)

    return dense_optimizer, sparse_optimizer
