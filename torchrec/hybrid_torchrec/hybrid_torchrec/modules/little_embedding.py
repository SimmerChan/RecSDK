#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Tuple, Dict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import os
import logging

from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
EmbeddingLocation,
ComputeDevice,
)

import torch
from torch import nn
import torch.distributed as dist
from torchrec import KeyedJaggedTensor, JaggedTensor
from hybrid_torchrec.distributed.batched_embedding_kernel import (
HybridSplitTableBatchedEmbeddingBagsCodegen,
)
from hybrid_torchrec.sparse.jagged_tensor_with_looup_helper import KeyedJaggedTensorWithLookHelper


class Awaitable:
    def __init__(self, a_function=None, *args):
        if a_function is not None:
            self.future = executor.submit(a_function, *args)
            self.result = None

    def wait(self):
        if self.result is None:
            self.result = self.future.result()
        return self.result


executor = ThreadPoolExecutor(6)


class PostInpuDistAwaitable(Awaitable):
    pass


class LookupAndOutputDistAwaitable(Awaitable):
    def __init__(self, post_awaitable, lookup_and_out_dist_function, *args):
        super().__init__()
        self.post_awaitable = post_awaitable
        self.lookup_and_out_dist_function = lookup_and_out_dist_function
        self.args = args

    def wait(self):
        post_result = self.post_awaitable.wait()
        return self.lookup_and_out_dist_function(post_result, *self.args)


# 不支持一表多查
@dataclass
class EmbeddingConfig:
    table_name: str
    num_embedding: int = 0
    embedding_dim: int = 0
    optimizer: OptimType = OptimType.EXACT_SGD
    world_size: int = 0


# 示例
class AllGatherEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fwd_pg, bwd_pg, embedding: torch.Tensor):
        ctx.fwd_gp = fwd_pg
        ctx.bwd_pg = bwd_pg
        ctx.embedding = embedding.data
        result_list = [torch.zeros_like(embedding) for i in range(2)]
        fwd_pg.allgather(result_list, embedding.data).wait()
        return tuple(result_list)

    @staticmethod
    def backward(ctx, *grad_output: torch.Tensor):
        grad_output = [g.data for g in grad_output]
        result = torch.zeros_like(grad_output[0])
        if grad_output[0].device != torch.device("cpu"):
            ctx.bwd_pg.reduce_scatter(result, grad_output).wait()
        else:
            # 模拟reduce scatter
            rank = ctx.bwd_pg.rank()
            grad_output_concat = torch.concat(grad_output)
            ctx.bwd_pg.allreduce(grad_output_concat, op=dist.reduce_op.SUM).wait()
            result = grad_output_concat.split(6)[rank]
        return None, None, result


class LookupContext:
    def __init__(self):
        self.fwd_pg
        self.bwd_pg
        self.communication_metrix = []


# 待实现
class AllGatherEmbeddings(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, embedding: torch.Tensor, context: LookupContext
    ) -> Tuple[torch.Tensor]:
        pass

    @staticmethod
    def backward(ctx, *grad_output: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, None]:
        pass


class HashEmbeddingModuleCollection(nn.Module):
    def __init__(self, configs=List[EmbeddingConfig], pipe_n_batch=6):
        super().__init__()
        self.fwd_pg = dist.new_group(backend="gloo")
        self.bwd_pg = dist.new_group(backend="gloo")
        self.rank = os.environ.get("RANK", 0)
        self.post_input_dist_module_dict: Dict[str, nn.Module] = (
            self.create_post_input_dist()
        )
        self.lookup_module = self.create_lookup()
        self.configs = configs

    def compute_context(self, fid: dict[JaggedTensor]) -> LookupContext:
        pass

    def create_post_input_dist(self) -> Dict[str, nn.Module]:
        return

    def create_lookups(self) -> nn.Module:
        embedding_specs = []
        table_names = []
        feature_table_map = []
        for idx, config in enumerate(self.configs):
            name = config.table_name
            num_embedding = config.num_embedding // config.world_size
            if self.rank == config.world_size - 1:
                num_embedding += config.num_embedding % config.world_size
            embedding_spec = (num_embedding, config.embedding_dim, EmbeddingLocation.DEVICE, ComputeDevice.NPU)
            embedding_specs.append(embedding_spec)
            table_names.append(name)
            feature_table_map.extend([idx] * 1)
        output_dtype = SparseType.FP32
        optimizer = config.optimizer
        optimizer_args = {"learning_rate": 0.01}
        pooling_mode = None
        device = torch.device("npu")
        lookup_module = HybridSplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[embedding_spec],
            feature_table_map=feature_table_map,
            output_dtype=output_dtype,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            pooling_mode=pooling_mode,
            device=device,
            table_names=table_names,
        )
        return lookup_module

    def do_post_input_dist(
        self, jt: JaggedTensor, feat_name: str, context: LookupContext
    ):
        return

    def do_lookup_and_post_dist(
        self, jt: JaggedTensor, feat_name: str, context: LookupContext
    ):
        return

    # 示例代码
    def fids2indices(self, fids: KeyedJaggedTensor):
        return fids

    # 示例代码
    def lookup(self, kjt: KeyedJaggedTensorWithLookHelper) -> torch.Tensor:
        return self.lookup_module(
            indices=kjt.values().long(),
            offsets=kjt.offsets().long(),
            hash_indices=kjt.hash_indices(),
            unique_indices=kjt.unique_indices(),
            unique_offsets=kjt.unique_offsets(),
            unique_inverse=kjt.unique_inverse()
        )

    def post_input_dist(self, jt: JaggedTensor, feat_name: str):
        indices = self.fids2indices(jt)
        return indices

    def lookup_and_post_dist(self, kjt: KeyedJaggedTensorWithLookHelper):
        embedding = self.lookup(kjt)
        result = AllGatherEmbedding().apply(self.fwd_pg, self.bwd_pg, embedding)
        return result

    def forward(self, kjt_list_each_rank: List[KeyedJaggedTensor]):
        jt_dict: Dict[str, JaggedTensor] = kjt_list_each_rank[self.rank].to_dict()
        awaitable_dict: Dict[str, LookupAndOutputDistAwaitable] = {}
        for feat_name in jt_dict.keys():
            post_awaitable = PostInpuDistAwaitable(
                self.post_input_dist, jt_dict[feat_name], feat_name
            )
            lookup_and_outdist_awaitable = LookupAndOutputDistAwaitable(
                post_awaitable, self.lookup_and_post_dist, feat_name
            )
            awaitable_dict[feat_name] = lookup_and_outdist_awaitable
        return awaitable_dict
