#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import logging
from typing import List

import torch
from torch.autograd.profiler import record_function

try:
    torch.ops.load_library(os.path.join(os.path.dirname(__file__), "libhybrid_cpp.so"))
except Exception as ex:
    logging.error(f"File libhybrid_cpp.so not found {ex}")


class HashMapBase(torch.nn.Module):
    def forward(self, ids: torch.Tensor, high_precison: bool) -> tuple[torch.Tensor]:
        pass


def ids2indices(ids, hashmap, high_precison):
    result = hashmap.ids2indices_unique(ids, high_precison)
    return result


class IdsMapperSimple(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ids2slot_dict = {}
        self.index = 0

    def forward(self, ids: torch.Tensor, high_precison) -> torch.Tensor:
        hash_indices_out = torch.empty_like(ids)
        hash_indices_out.resize(ids.numel())
        n = ids.shape[0]
        for i in range(n):
            k = ids[i].item()
            if k in self.ids2slot_dict:
                hash_indices_out[i] = self.ids2slot_dict[k]
            else:
                self.ids2slot_dict[k] = self.index
                hash_indices_out[i] = self.index
                self.index += 1
        unique, unique_inverse = torch.unique(hash_indices_out, return_inverse=True)
        return hash_indices_out, unique, unique_inverse


class IdsMapper(HashMapBase):
    """
    This class is primarily used for managing global ids.
    Its core functionality is to convert global ids into indices,
    which represent offsets in the embedding table stored on the NPU device.
    Additionally, it provides functionality for evicting ids,
    exporting all ids along with their corresponding indices,
    retrieving timestamps for the ids, and returning the current count
    of ids stored in the IdsMapper.
    """

    def __init__(self, n):
        super().__init__()
        self.ids_mapper = torch.classes.hybrid.IdsMapper(n)
        self.n = n

    def forward(self, ids: torch.Tensor, high_precison: bool):
        with record_function("## ids2indices ##"):
            result, unique, unique_inverse = self.ids_mapper.ids2indices_unique(
                ids, high_precison
            )
            return result, unique, unique_inverse

    def ids2indices_unique_out(
        self,
        ids: torch.Tensor,
        hashIndices: torch.Tensor,
        offset: torch.Tensor,
        unique: torch.Tensor,
        uniqueInverse: torch.Tensor,
        uniqueOffset: List[int],
        tensorI: int,
    ):
        self.ids_mapper.ids2indices_unique_out(
            ids, hashIndices, offset, unique, uniqueInverse, uniqueOffset, tensorI
        )
