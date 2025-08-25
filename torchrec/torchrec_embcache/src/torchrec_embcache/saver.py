#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging

import torch.distributed as dist
import torch.nn

from torchrec_embcache.distributed.embedding_bag import EmbCacheShardedEmbeddingBagCollection
from torchrec_embcache.distributed.embedding import EmbCacheShardedEmbeddingCollection

_SAVE_PATH_MIN_LEN = 1
_SAVE_PATH_MAX_LEN = 1024


class Saver:
    def __init__(self, rank: int = None):
        if rank is None:
            if dist.is_initialized():
                rank = dist.get_rank()
                logging.warning("Param rank id is None and distributed model has been initialized,"
                                " get rank by dist.get_rank() is:%d", rank)
            else:
                raise ValueError("param `rank` must not be None when torch.distributed.is_initialized() is False.")
        else:
            if not (isinstance(rank, int) and not isinstance(rank, bool) and rank >= 0):
                raise ValueError("param rank must an integer and greater or equal than 0.")
        self.rank: int = rank
        self.cache_module = []

    def save(self, module: torch.nn.Module, path: str) -> None:
        if not isinstance(module, torch.nn.Module):
            raise ValueError(f"param `module` must an instance of torch.nn.Module, but got:{type(module)}")
        if not (isinstance(path, str) and _SAVE_PATH_MIN_LEN <= len(path) <= _SAVE_PATH_MAX_LEN):
            raise ValueError(f"param `path` must be str and length in [{_SAVE_PATH_MIN_LEN}, {_SAVE_PATH_MAX_LEN}].")

        self.cache_module.clear()
        self._find_all_embed_cache_instance(module)
        self._check_emb_cache_instance_len()

        logging.info("In save scene, cache_module info:%s", self.cache_module)
        for mod in self.cache_module:
            logging.info("In save scene, embcache_mgr info:%s", mod.embcache_mgr)
            codegen = mod.get_batched_embedding_kernels()[0][0]
            momentum_list = [momentum.detach().to("cpu") for momentum in codegen.get_momentum()] 
            mod.embcache_mgr.embedding_to_host(codegen.weights_dev.detach().to("cpu"), momentum_list)
            mod.embcache_mgr.save(path, self.rank)

    def _check_emb_cache_instance_len(self):
        if len(self.cache_module) == 0:
            raise ValueError("param `module` must has at least one child module which "
                             "type is EmbCacheShardedEmbeddingBagCollection or EmbCacheShardedEmbeddingBagCollection.")

    def load(self, module: torch.nn.Module, path: str) -> None:
        self.cache_module.clear()
        self._find_all_embed_cache_instance(module)
        self._check_emb_cache_instance_len()
        for mod in self.cache_module:
            mod.embcache_mgr.load(path, self.rank)

    def _find_all_embed_cache_instance(self, module):
        for _, child in module.named_children():
            if (isinstance(child, EmbCacheShardedEmbeddingBagCollection)
                    or isinstance(child, EmbCacheShardedEmbeddingCollection)):
                self.cache_module.append(child)
            self._find_all_embed_cache_instance(child)

