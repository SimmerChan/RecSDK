#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

__all__ = ["logger", "loader", "make_singleton", "tf_npu_ops", "gen_npu_cpu_ops", "get_tfa_op"]

from mxrec.python.utils.log import LoggingProxy as logger
from mxrec.python.utils import loader
from mxrec.python.utils.tfa import tf_npu_ops, gen_npu_cpu_ops, get_tfa_op
from mxrec.python.utils.decorator import make_singleton
