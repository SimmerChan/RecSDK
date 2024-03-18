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

import numpy as np
import tensorflow as tf
from mx_rec.util.log import logger


class CalculateAucHook(tf.train.SessionRunHook):
    def __init__(self, mode):
        np.set_printoptions(suppress=True)
        np.set_printoptions(linewidth=400)
        self.mode = mode
        self.best_auc = 0.0

    def before_run(self, run_context):
        """返回SessionRunArgs和session run一起跑"""
        steps = tf.get_collection('hook_steps')
        prediction = None
        label = None
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            prediction = tf.get_collection('train_prediction')
            label = tf.get_collection('train_label')
        elif self.mode == tf.estimator.ModeKeys.EVAL:
            prediction = tf.get_collection('eval_prediction')
            label = tf.get_collection('eval_label')
        elif self.mode == tf.estimator.ModeKeys.PREDICT:
            prediction = tf.get_collection('predict_prediction')
            label = tf.get_collection('predict_label')
        return tf.train.SessionRunArgs(fetches=[prediction, label, steps])

    def after_run(self, run_context, run_values):
        prediction, label, steps = run_values.results
        from sklearn.metrics import roc_auc_score
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            if steps[0] % 10 == 0:
                test_auc = roc_auc_score(label[0], prediction[0])
                self.best_auc = max(self.best_auc, test_auc)
                logger.info("train_steps: %d;train_auc: %s;best_auc: %s" % (steps[0], test_auc, self.best_auc))
        elif self.mode == tf.estimator.ModeKeys.EVAL:
            logger.info("eval_auc: %s", roc_auc_score(label[0], prediction[0]))
        elif self.mode == tf.estimator.ModeKeys.PREDICT:
            logger.info("predict_auc: %s", roc_auc_score(label[0], prediction[0]))
