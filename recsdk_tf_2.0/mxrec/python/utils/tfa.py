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

import sysconfig

import tensorflow as tf

if tf.__version__.startswith("1"):  # pragma: no cover
    from npu_bridge.estimator import npu_ops as tf_npu_ops
    from npu_bridge.npu_cpu.npu_cpu_ops import gen_npu_cpu_ops
else:  # pragma: no cover
    from npu_device.compat.v1.estimator import npu_ops as tf_npu_ops
    from npu_device.compat.v1.npu_cpu.npu_cpu_ops import gen_npu_cpu_ops


def get_tfa_op():
    return tf.load_op_library(f"{sysconfig.get_path('purelib')}/mxrec/librec/libasc_ops.so")