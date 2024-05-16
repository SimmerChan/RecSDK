import os
from typing import List

import tensorflow as tf
from tensorflow import Tensor, Operation
from mx_rec.util.log import logger
from mx_rec.util.initialize import ConfigInitializer, init, terminate_config_initializer
from mx_rec.graph import modify_graph_and_start_emb_cache

from datasets import gen_tf_dataset
from models import MatrixFactorization
from utils import GlobalConfig, get_sess_config

tf.compat.v1.disable_eager_execution()


def train() -> None:
    loss: Tensor = None
    train_ops: List[Operation] = None

    graph = tf.compat.v1.Graph()
    with graph.as_default():
        cfg = GlobalConfig()

        dataset = gen_tf_dataset(cfg)
        dataset = dataset.prefetch(100)
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        batch = iterator.get_next()

        model = MatrixFactorization(cfg, iterator)
        loss = model.forward(batch["user_ids"], batch["item_ids"], batch["labels"], is_training=True)
        train_ops = model.backward(loss)

    modify_graph_and_start_emb_cache(full_graph=graph, dump_graph=True)

    with tf.compat.v1.Session(graph=graph, config=get_sess_config()) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        iter_initializer = ConfigInitializer.get_instance().train_params_config.get_initializer(is_training=True)
        sess.run(iter_initializer)

        while True:
            try:
                sess.run(fetches=[loss, train_ops])
            except tf.errors.OutOfRangeError:
                logger.info("End of model training.")
                break

        saver = tf.compat.v1.train.Saver()
        saver.save(sess, save_path="./ckpts/")


def infer() -> None:
    loss: Tensor = None

    graph = tf.compat.v1.Graph()
    with graph.as_default():
        cfg = GlobalConfig()

        dataset = gen_tf_dataset(cfg)
        dataset = dataset.prefetch(100)
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        batch = iterator.get_next()

        model = MatrixFactorization(cfg, iterator)
        loss = model.forward(batch["user_ids"], batch["item_ids"], batch["labels"], is_training=False)

    modify_graph_and_start_emb_cache(full_graph=graph, dump_graph=True)

    with tf.compat.v1.Session(graph=graph, config=get_sess_config()) as sess:
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, save_path="./ckpts/")

        iter_initializer = ConfigInitializer.get_instance().train_params_config.get_initializer(is_training=False)
        sess.run(iter_initializer)

        while True:
            try:
                sess.run(fetches=[loss])
            except tf.errors.OutOfRangeError:
                logger.info("End of model inference.")
                break


if __name__ == "__main__":
    init(use_dynamic=True, use_dynamic_expansion=True)

    use_mode = os.getenv("USE_MODE")
    if use_mode == "train":
        train()
    elif use_mode == "infer":
        infer()
    else:
        raise ValueError(f"got invalid `USE_MODE` env var {use_mode}")

    terminate_config_initializer()
    logger.info("Demo done!")
