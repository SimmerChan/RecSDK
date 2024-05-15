import time
from typing import List

import tensorflow as tf
from tensorflow import Tensor, Operation
from mx_rec.util.log import logger
from mx_rec.util.initialize import ConfigInitializer, init, terminate_config_initializer
from mx_rec.graph import modify_graph_and_start_emb_cache

from utils import ModelConfig, get_sess_config
from datasets import gen_tf_dataset
from models import MatrixFactorization

tf.compat.v1.disable_eager_execution()

if __name__ == "__main__":
    init(use_dynamic=True, use_dynamic_expansion=True)

    loss: Tensor = None
    train_ops: List[Operation] = None

    graph = tf.compat.v1.Graph()
    with graph.as_default():
        cfg = ModelConfig()

        dataset = gen_tf_dataset(cfg)
        dataset = dataset.prefetch(100)
        iterator = dataset.make_initializable_iterator()
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
                sess.run(fetches=[train_ops, loss])
            except tf.errors.OutOfRangeError:
                logger.info("End of model training.")
                break

    terminate_config_initializer()
    logger.info("Demo done!")
