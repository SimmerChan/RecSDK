import numpy as np
from typing import Callable, Dict

import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import DatasetV1Adapter
from mx_rec.util.communication import hccl_ops

from utils import GlobalConfig


def gen_tf_dataset(cfg: GlobalConfig, batch_number: int = 100) -> DatasetV1Adapter:
    dataset = tf.compat.v1.data.Dataset.from_generator(
        generator=_gen_rand_data(cfg, batch_number=batch_number),
        output_types={"item_ids": tf.int32, "user_ids": tf.int32, "labels": tf.int32},
        output_shapes={
            "user_ids": tf.TensorShape([cfg.batch_size, cfg.user_feat_cnt]),
            "item_ids": tf.TensorShape([cfg.batch_size, cfg.item_feat_cnt]),
            "labels": tf.TensorShape([cfg.batch_size]),
        },
    )

    rank_size = hccl_ops.get_rank_size()
    rank_id = hccl_ops.get_rank_id()
    if rank_size > 1:
        dataset = dataset.shard(rank_size, rank_id)

    return dataset


def _gen_rand_data(cfg: GlobalConfig, batch_number: int) -> Callable[[], Dict[str, np.ndarray]]:
    def data_generator():
        i = 0
        while i < batch_number:
            user_ids = np.random.randint(0, cfg.user_range, (cfg.batch_size, cfg.user_feat_cnt))
            item_ids = np.random.randint(0, cfg.item_range, (cfg.batch_size, cfg.item_feat_cnt))
            labels = np.random.randint(0, 2, (cfg.batch_size))

            i += 1
            yield {"user_ids": user_ids, "item_ids": item_ids, "labels": labels}

    return data_generator
