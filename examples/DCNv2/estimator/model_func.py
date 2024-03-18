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

from calculate_hook import CalculateAucHook
from config import Config
from dcnv2 import MyModel
from optim import get_dense_and_sparse_optimizer
from optim import get_train_op
from utils import FeatureSpecIns

from mx_rec.core.embedding import create_table
from mx_rec.core.embedding import sparse_lookup


def eval_estimator_spec(summary_dict, hook_list):
    tf.add_to_collection("eval_prediction", summary_dict.get("prediction"))
    tf.add_to_collection("eval_label", summary_dict.get("label"))
    return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.EVAL,
                                      loss=summary_dict.get("loss"),
                                      evaluation_hooks=hook_list
                                      )


def train_estimator_spec(cfg, summary_dict, hook_list, train_op):
    # 打印频率等于every_n_iter*iterations_per_loop
    logging_hook = tf.train.LoggingTensorHook({
        "train_loss": summary_dict.get("loss"),
        "dense_lr": cfg.learning_rate[0],
        "sparse_lr": cfg.learning_rate[1],
        "step": cfg.global_step,
    }, every_n_iter=100)
    hook_list.append(logging_hook)
    tf.summary.scalar("train_loss", summary_dict.get("loss"))
    tf.add_to_collection("train_prediction", summary_dict.get("prediction"))
    tf.add_to_collection("train_label", summary_dict.get("label"))
    tf.add_to_collection("hook_steps", cfg.global_step)
    return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN,
                                      loss=summary_dict.get("loss"),
                                      train_op=train_op,
                                      training_hooks=hook_list)


def predict_estimator_spec(summary_dict, hook_list):
    predict_dict = {}
    predict_dict["pred"] = summary_dict.get("prediction")
    predict_dict["label"] = summary_dict.get("label")
    export_output = {"predictor": tf.estimator.export.PredictOutput(predict_dict)}
    tf.add_to_collection("predict_prediction", summary_dict.get("prediction"))
    tf.add_to_collection("predict_label", summary_dict.get("label"))
    return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT,
                                      predictions=predict_dict,
                                      export_outputs=export_output,
                                      prediction_hooks=hook_list)


def get_model_fn():
    def model_fn(features, labels, mode, params):
        cfg = Config(params)
        model = MyModel()

        dense_optimizer, sparse_optimizer = get_dense_and_sparse_optimizer(cfg)
        emb_initializer = tf.compat.v1.truncated_normal_initializer(stddev=0.05, seed=cfg.seed) \
            if cfg.cache_mode != "HBM" or params.use_dynamic_expansion else \
            tf.compat.v1.variance_scaling_initializer(mode="fan_avg", distribution="normal", seed=cfg.seed)
        sparse_hashtable = create_table(key_dtype=cfg.key_type,
                                        dim=tf.TensorShape([cfg.emb_dim]),
                                        name="sparse_embeddings_table",
                                        emb_initializer=emb_initializer,
                                        optimizer_list=[sparse_optimizer.optimizer],
                                        **cfg.get_emb_table_cfg()
                                        )

        if cfg.modify_graph:
            feature_spec_list = [features["sparse_feature"]]
        else:
            if mode == tf.estimator.ModeKeys.TRAIN:
                feature_spec_list = FeatureSpecIns.get_instance().get_train_feature_spec_list()
            else:
                feature_spec_list = FeatureSpecIns.get_instance().get_eval_feature_spec_list()
        embedding = sparse_lookup(sparse_hashtable,
                                  feature_spec_list[0],
                                  cfg.send_count,
                                  dim=None,
                                  is_train=(mode == tf.estimator.ModeKeys.TRAIN),
                                  batch=features,
                                  name="sparse_embeddings",
                                  modify_graph=cfg.modify_graph)

        loss, prediction = model.build_model(embedding, features["dense_feature"], features["label"])
        summary_dict = {"loss": loss, "prediction": prediction, "label": features["label"]}
        hook_list = [CalculateAucHook(mode)]
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = get_train_op(cfg, summary_dict.get("loss"), dense_optimizer, sparse_optimizer)
            return train_estimator_spec(cfg, summary_dict, hook_list, train_op)
        if mode == tf.estimator.ModeKeys.PREDICT:
            return predict_estimator_spec(summary_dict, hook_list)
        if mode == tf.estimator.ModeKeys.EVAL:
            return eval_estimator_spec(summary_dict, hook_list)

        raise ValueError("mode is wrong.")

    return model_fn
