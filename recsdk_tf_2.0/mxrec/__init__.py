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

__version__ = "6.0.T200"
__all__ = [
    "version",
    "__version__",
    "init",
    "get_rank_id",
    "get_rank_size",
    "get_local_rank_size",
    "get_embedding_table",
    "embedding_lookup",
    "get_sparse_embedding",
    "get_init_hashtable_op",
    "get_existing_tables",
    "AdamWOptimizer",
    "EmbeddingTableSaver",
]


from mxrec.python.communication import get_local_rank_size, get_rank_id, get_rank_size
from mxrec.python.embedding import (
    embedding_lookup,
    get_embedding_table,
    get_existing_tables,
    get_init_hashtable_op,
    get_sparse_embedding,
)
from mxrec.python.initializer import init
from mxrec.python.optimizer import AdamWOptimizer, patch_for_update_op
from mxrec.python.training import EmbeddingTableSaver


def version():
    return __version__


def _patch_for_mxrec():
    patch_for_update_op()


_patch_for_mxrec()
