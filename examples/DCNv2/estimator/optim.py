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

from npu_bridge.hccl import hccl_ops
from delay_loss_scale import DenseLossScaleOptimizer, SparseLossScaleOptimizer

from mx_rec.constants.constants import ASCEND_SPARSE_LOOKUP_LOCAL_EMB, ASCEND_SPARSE_LOOKUP_UNIQUE_KEYS
from mx_rec.optimizers.lazy_adam import create_hash_optimizer
from mx_rec.optimizers.lazy_adam_by_addr import create_hash_optimizer_by_address
from mx_rec.util.communication.hccl_ops import get_rank_size
from mx_rec.util.initialize import ConfigInitializer
from mx_rec.util.variable import get_dense_and_sparse_variable


def get_dense_and_sparse_optimizer(cfg):
    dense_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=cfg.learning_rate[0])
    use_dynamic_expansion = ConfigInitializer.get_instance().use_dynamic_expansion
    if use_dynamic_expansion:
        sparse_optimizer = create_hash_optimizer_by_address(learning_rate=cfg.learning_rate[1])
    else:
        sparse_optimizer = create_hash_optimizer(learning_rate=cfg.learning_rate[1])
    sparse_optimizer = SparseLossScaleOptimizer(sparse_optimizer, 65536)

    dense_optimizer = DenseLossScaleOptimizer(dense_optimizer, 65536)

    return dense_optimizer, sparse_optimizer


def get_train_op_list(loss, cfg, dense_optimizer, sparse_optimizer):
    train_ops_list = []
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    dense_variables, sparse_variables = get_dense_and_sparse_variable()
    with tf.control_dependencies(update_ops):
        # do dense optimization
        grads = dense_optimizer.compute_gradients(loss, var_list=dense_variables)
        dense_grads = grads[:len(dense_variables)]
        avg_grads = []
        for grad, var in dense_grads:
            if get_rank_size() > 1:
                grad = hccl_ops.allreduce(grad, "sum") if grad is not None else None
            if grad is not None:
                avg_grads.append((grad / 8.0, var))
        # apply gradients: update variables
        train_ops_list.append(dense_optimizer.apply_gradients(avg_grads, name="dense_optimizer"))
        if cfg.use_dynamic_expansion:
            train_address_list = tf.compat.v1.get_collection(ASCEND_SPARSE_LOOKUP_UNIQUE_KEYS)
            train_emb_list = tf.compat.v1.get_collection(ASCEND_SPARSE_LOOKUP_LOCAL_EMB)
            # do sparse optimization by addr
            sparse_grads = sparse_optimizer.compute_gradients(loss, train_emb_list)  # local_embedding
            grads_and_vars = [(grad, address) for grad, address in zip(sparse_grads, train_address_list)]
            train_ops_list.append(sparse_optimizer.apply_gradients(grads_and_vars, name="hashtable_optimizer"))
        else:
            # do sparse optimization
            sparse_grads = sparse_optimizer.compute_gradients(loss, sparse_variables)
            grads_and_vars = [(grad, variable) for grad, variable in zip(sparse_grads, sparse_variables)]
            train_ops_list.append(sparse_optimizer.apply_gradients(grads_and_vars, name="sparse_optimizer"))

    global_step_op = tf.assign_add(tf.train.get_global_step(), 1)
    train_ops_list.append(global_step_op)
    # 动态学习率更新
    train_ops_list.extend([cfg.global_step.assign(cfg.global_step + 1), cfg.learning_rate[0], cfg.learning_rate[1]])
    with tf.control_dependencies(train_ops_list):
        cfg.learning_rate = [cfg.learning_rate[0], cfg.learning_rate[1]]
        tf.summary.scalar("learning_rate", cfg.learning_rate[0])
        tf.summary.scalar("learning_rate", cfg.learning_rate[1])
    return train_ops_list


def get_train_op(cfg, losses, dense_optimizer, sparse_optimizer):
    train_ops = []
    op_list = get_train_op_list(losses, cfg, dense_optimizer, sparse_optimizer)
    train_ops.append(['_train', tf.group(*op_list)])
    ops = [losses] + [train_op[1] for train_op in train_ops]
    return tf.group(*ops)
