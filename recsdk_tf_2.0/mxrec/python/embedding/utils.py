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

import tensorflow as tf

from mxrec.python.embedding.table.base_emb_table import BaseEmbTable
from mxrec.python.embedding.lookup.base_lookup import BaseLookup


def get_table_ins_by_local_embedding(local_embedding: tf.Tensor) -> BaseEmbTable:
    local_emb_to_table_ins = BaseLookup.get_local_emb_to_table_ins()
    if local_embedding not in local_emb_to_table_ins:
        raise KeyError(f"the local embedding {local_embedding} does not exist in {local_emb_to_table_ins}")
    return local_emb_to_table_ins.get(local_embedding)
