from typing import Tuple, List
from abc import ABCMeta, abstractmethod

import tensorflow as tf
from tensorflow import Tensor, Operation
from tensorflow.python.data import Iterator
from mx_rec.core import sparse_lookup, create_table
from mx_rec.core.emb.base_sparse_embedding import BaseSparseEmbedding
from mx_rec.constants.constants import ASCEND_SPARSE_LOOKUP_LOCAL_EMB, ASCEND_SPARSE_LOOKUP_ID_OFFSET
from mx_rec.optimizers.gradient_descent_by_addr import create_hash_optimizer_by_addr
from mx_rec.util.variable import get_dense_and_sparse_variable
from mx_rec.util.communication.hccl_ops import get_rank_size

from utils import GlobalConfig, hccl_ops


class BaseModel(metaclass=ABCMeta):
    def __init__(self, cfg: GlobalConfig, iterator: Iterator) -> None:
        self.user_feat_cnt = cfg.user_feat_cnt
        self.item_feat_cnt = cfg.item_feat_cnt

        self.user_table_dim = cfg.user_table_dim
        self.item_table_dim = cfg.item_table_dim

        self.learning_rate = cfg.learning_rate

        self._iterator = iterator

    @abstractmethod
    def _build_tables(self) -> Tuple[BaseSparseEmbedding, BaseSparseEmbedding]:
        pass

    @abstractmethod
    def forward(self, user_ids: Tensor, item_ids: Tensor, label: Tensor, is_training: bool) -> Tensor:
        pass

    @abstractmethod
    def backward(self, loss: Tensor) -> List[Operation]:
        pass


class MatrixFactorization(BaseModel):
    def __init__(self, cfg: GlobalConfig, iterator: Iterator) -> None:
        super().__init__(cfg, iterator)

    def _build_tables(self) -> Tuple[BaseSparseEmbedding, BaseSparseEmbedding]:
        user_table = create_table(
            key_dtype=tf.int32,
            dim=tf.TensorShape([self.user_table_dim]),
            name="user_table",
            emb_initializer=tf.compat.v1.truncated_normal_initializer(),
        )
        item_table = create_table(
            key_dtype=tf.int32,
            dim=tf.TensorShape([self.item_table_dim]),
            name="item_table",
            emb_initializer=tf.compat.v1.truncated_normal_initializer(),
        )

        return (user_table, item_table)

    def forward(self, user_ids: Tensor, item_ids: Tensor, labels: Tensor, is_training: bool) -> Tensor:
        (user_table, item_table) = self._build_tables()

        user_embs = sparse_lookup(hashtable=user_table, ids=user_ids, is_train=is_training, modify_graph=True)
        item_embs = sparse_lookup(hashtable=item_table, ids=item_ids, is_train=is_training, modify_graph=True)
        biases = tf.compat.v1.Variable(tf.zeros(shape=user_embs.shape, dtype=tf.float32), name="biases")

        preds = tf.reduce_sum(tf.multiply(user_embs, item_embs) + biases, axis=[1, 2])
        labels = tf.cast(labels, dtype=tf.float32)
        losses = tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=preds)

        return tf.reduce_mean(losses)

    def backward(self, loss: Tensor) -> List[Operation]:
        train_ops: List[Operation] = []

        dense_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        sparse_optimizer = create_hash_optimizer_by_addr(learning_rate=self.learning_rate)

        dense_vars, _ = get_dense_and_sparse_variable()
        dense_grads_and_vars = dense_optimizer.compute_gradients(loss, var_list=dense_vars)
        avg_grad_and_vars = []

        for grad, var in dense_grads_and_vars:
            if get_rank_size() > 1:
                grad = hccl_ops.allreduce(grad, "sum") if grad is not None else None
            if grad is not None:
                avg_grad_and_vars.append((grad, var))
        train_ops.append(dense_optimizer.apply_gradients(avg_grad_and_vars))

        train_embs = tf.compat.v1.get_collection(ASCEND_SPARSE_LOOKUP_LOCAL_EMB)
        train_addrs = tf.compat.v1.get_collection(ASCEND_SPARSE_LOOKUP_ID_OFFSET)

        sparse_grads = tf.gradients(loss, train_embs)
        sparse_grads_and_vars = [(grad, addr) for grad, addr in zip(sparse_grads, train_addrs)]
        train_ops.append(sparse_optimizer.apply_gradients(sparse_grads_and_vars))

        return train_ops
