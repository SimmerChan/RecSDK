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

__all__ = [
    "EmbeddingResource",
    "get_embedding_table",
    "embedding_lookup",
    "get_sparse_embedding",
    "get_init_hashtable_op",
    "get_table_ins_by_local_embedding",
    "get_existing_tables",
]

from mxrec.python.embedding.embedding import (
    EmbeddingResource,
    embedding_lookup,
    get_embedding_table,
    get_existing_tables,
    get_init_hashtable_op,
    get_sparse_embedding,
)
from mxrec.python.embedding.utils import get_table_ins_by_local_embedding
