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

import time
import tensorflow as tf

import mx_rec.util as mxrec_util
from mx_rec.util.log import logger


class CalcQpsHook(tf.estimator.SessionRunHook):
    def __init__(self, batch_size, iterations_per_loop):
        self.start_time = None
        self.rank_size = mxrec_util.communication.hccl_ops.get_rank_size()
        self.batch_size = batch_size
        self.iterations_per_loop = iterations_per_loop
        self.step_cnt = 0

    def before_run(self, run_context):
        self.start_time = time.time()

    def after_run(self, run_context, run_values):
        self.step_cnt += 1
        end_time = time.time()
        cost_time = end_time - self.start_time
        qps = (1 / cost_time) * self.rank_size * self.batch_size * self.iterations_per_loop
        logger.info(f"step: {self.step_cnt * self.iterations_per_loop};"
                    f" current sess cost time: {cost_time:.10f}, current QPS: {qps}")
