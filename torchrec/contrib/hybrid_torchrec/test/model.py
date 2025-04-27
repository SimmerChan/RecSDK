#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict
from dataset import Batch
import torch
from torchrec import (
    EmbeddingBagCollection,
    EmbeddingCollection,
    KeyedJaggedTensor,
)
from hybrid_torchrec import (
    HashEmbeddingBagCollection,
    HashEmbeddingCollection,
)
from hybrid_torchrec.distributed.embeddingbag import HybridShardedEmbeddingBagCollection
from hybrid_torchrec.distributed.embedding import HybridShardedEmbeddingCollection

def permute_values_ebc(kjt: KeyedJaggedTensor, feature_num) -> torch.Tensor:
    keys_nums = feature_num
    values = []
    jt_dict = kjt.to_dict()
    for k in range(keys_nums):
        k = f"feat{k}"
        jt = jt_dict[k]
        values.append(jt)
    values = torch.concat(values, dim=1)
    return values

# ec和ebc查询结果返回数据类型不一样
def permute_values_ec(result: Dict, feature_num) -> torch.Tensor:
    keys_nums = feature_num
    values = []
    for k in range(keys_nums):
        k = f"feat{k}"
        jt = result[k].values()
        values.append(jt)
    values = torch.concat(values, dim=1)
    return values

class Model(torch.nn.Module):
    def __init__(self, module, feature_num):
        super().__init__()
        self._module = module
        self.feature_num = feature_num

    @property
    def _module_type(self):
        if isinstance(self._module,(EmbeddingBagCollection, HashEmbeddingBagCollection, HybridShardedEmbeddingBagCollection)):
            return "ebc"
        elif isinstance(self._module,(EmbeddingCollection, HashEmbeddingCollection, HybridShardedEmbeddingCollection)):
            return "ec"
        else:
            raise ValueError("modules must be EmbeddingCollection or EmbeddingBagCollection")

    @property
    def ebc(self):
        self._ebc = self._module if self._module_type == "ebc" else None
        return self._ebc
    @property
    def ec(self):
        self._ec = self._module if self._module_type == "ec" else None
        return self._ec

    def forward(self, batch: Batch):
        result = self._module(batch.sparse_features)
        if self._module_type == "ebc":
            result = permute_values_ebc(result, self.feature_num)
        elif self._module_type == "ec":
            result = permute_values_ec(result, self.feature_num)
        loss = result.sum()
        return loss, result