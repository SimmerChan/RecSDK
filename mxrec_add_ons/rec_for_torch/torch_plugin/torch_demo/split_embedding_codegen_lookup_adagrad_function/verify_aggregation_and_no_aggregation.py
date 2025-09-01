#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

import logging
import random
import sysconfig
from collections import defaultdict
from dataclasses import dataclass

import pytest
import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import SplitTableBatchedEmbeddingBagsCodegen
from hybrid_torchrec.distributed.batched_embedding_kernel import HybridSplitTableBatchedEmbeddingBagsCodegen
from torch.optim import Adam, Adagrad, SGD

import torchrec
from torchrec import JaggedTensor, KeyedJaggedTensor, PoolingType, ComputeDevice


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

logging.getLogger().setLevel(logging.INFO)
DEVICEID = "npu:0"
EPOCH = 1
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

TORCH_POOLING_MODE_TO_FBGEMM = {
    PoolingType.SUM: PoolingMode.SUM,
    PoolingType.MEAN: PoolingMode.MEAN,
    PoolingType.NONE: PoolingMode.NONE,
}
TORCH_OPTIMIZER_TO_FBGEMM = {
    Adam: EmbOptimType.ADAM,
    Adagrad: EmbOptimType.EXACT_ADAGRAD,
    SGD: EmbOptimType.EXACT_SGD
}
OPTIMIZER_PARAM = {
    Adam: dict(lr=0.01),
    Adagrad: dict(lr=0.01, eps=1.0e-8),
    SGD: dict(lr=0.01),
}


@dataclass
class LookupParams:
    tables: list[list[int]]
    mutile_hots: list[int]
    batch_size: int
    pooling_mode: PoolingMode
    unique: bool
    optim: torch.optim
    feature_map: list[int] = None


def create_data(params):
    indices_test = []
    offsets_test = []
    jt_lst = []

    for ind, tid in enumerate(params.feature_map):
        table = params.tables[tid]  # seq,dim (200, 32)

        indices = torch.randint(0, table[0], (params.batch_size * params.mutile_hots[ind],)).to(torch.int64)
        indices_test.append(indices)

        offsets = torch.Tensor([params.mutile_hots[ind] for _ in range(params.batch_size)]).to(torch.int64)
        offsets_test.append(offsets)

        jt_lst.append(JaggedTensor(values=indices, lengths=offsets))

    return indices_test, offsets_test, jt_lst


def generate_unique(jt_lst, feature_map):
    unique_indices = []
    unique_inverse = []
    unique_offset = []
    start = 0

    jt_values = defaultdict(list)
    for ind, tid in enumerate(feature_map):
        jt_values[tid].append(jt_lst[ind].values())

    for key in jt_values:
        jt = torch.cat(jt_values[key])
        unique_indice, inverse = torch.unique(jt, return_inverse=True)
        unique_indices.append(unique_indice)
        unique_inverse.append(inverse)
        unique_offset.extend(len(jt_values[key]) * [start])
        start += unique_indice.shape[0]

    unique_offset.extend([start])
    return unique_indices, unique_inverse, unique_offset


def look_table(indices, offsets, jt_lst, tbe, params):
    if params.unique:
        unique_indices, unique_inverse, unique_offset = generate_unique(jt_lst, params.feature_map)
        unique_indices = torch.cat(unique_indices).to(DEVICEID).to(torch.int64)
        unique_inverse = torch.cat(unique_inverse).to(DEVICEID).to(torch.int64)
        unique_offset = torch.Tensor(unique_offset).to(DEVICEID).to(torch.int64)
        kwargs = dict(unique_indices=unique_indices, unique_offset=unique_offset, unique_inverse=unique_inverse)
    else:
        kwargs = dict()

    output = tbe(indices, offsets, **kwargs)  # bs,dim
    loss = torch.sum(output ** 2 / 2)
    loss.backward()
    return tbe.weights_dev


def concat_tensors_by_category(a):
    num_categories = len(a[0])
    result = []
    for category_idx in range(num_categories):
        tensors = [sublist[category_idx] for sublist in a]
        result.append(torch.cat(tensors))

    return result


def verify_grad_aggregation(params):
    torch.npu.set_device(DEVICEID)

    embedding_specs = [
        (num_embeddings, embedding_dim, EmbeddingLocation.DEVICE, ComputeDevice.NPU)
        for (num_embeddings, embedding_dim) in params.tables]

    if params.unique:
        ebc_class = HybridSplitTableBatchedEmbeddingBagsCodegen
    else:
        ebc_class = SplitTableBatchedEmbeddingBagsCodegen

    accumulate_step = 5

    tbe_grad_aggregation = ebc_class(
        embedding_specs=embedding_specs,
        optimizer=TORCH_OPTIMIZER_TO_FBGEMM[params.optim],
        device=torch.device(DEVICEID),
        pooling_mode=TORCH_POOLING_MODE_TO_FBGEMM[params.pooling_mode],
        feature_table_map=params.feature_map,
        use_accumulate=True,
        accumulate_step=accumulate_step
    )

    total_size = sum([num_embeddings * embedding_dim for (num_embeddings, embedding_dim) in params.tables])
    weights_test = torch.randn(total_size).to(torch.float32)
    weights_test = weights_test.to(DEVICEID)

    tbe_grad_aggregation.weights_dev = torch.nn.Parameter(weights_test.clone()).to(DEVICEID)

    tbe_no_grad_aggregation = ebc_class(
        embedding_specs=embedding_specs,
        optimizer=TORCH_OPTIMIZER_TO_FBGEMM[params.optim],
        device=torch.device(DEVICEID),
        pooling_mode=TORCH_POOLING_MODE_TO_FBGEMM[params.pooling_mode],
        feature_table_map=params.feature_map,
        use_accumulate=False)

    tbe_no_grad_aggregation.weights_dev = torch.nn.Parameter(weights_test.clone()).to(DEVICEID)

    for i in range(2):
        all_idx = []
        all_offsets = []
        for step in range(accumulate_step):  # TODO
            indices_test, offsets_test, jt_lst = create_data(params)
            all_idx.append(indices_test)
            all_offsets.append(offsets_test)

            indices_test = torch.cat(indices_test).to(torch.int64)

            offsets_test = torch.cat(offsets_test).to(torch.int64)
            offsets_test = torch.cat([torch.Tensor([0]), offsets_test]).to(torch.int64)
            offsets_test = torch.cumsum(offsets_test, dim=0)  # [0, 1, 2, 3, 4, 5, 6, 7, 8]

            indices_test = indices_test.to(DEVICEID)
            offsets_test = offsets_test.to(DEVICEID)
            weights_grad_aggregation = look_table(indices_test, offsets_test, jt_lst, tbe_grad_aggregation, params)

        all_jt_lst = []
        all_idx = concat_tensors_by_category(all_idx)

        all_offsets = concat_tensors_by_category(all_offsets)
        for i in range(len(params.tables)):
            all_jt_lst.append(JaggedTensor(values=all_idx[i], lengths=all_offsets[i]))

        all_idx = torch.cat(all_idx)

        all_offsets = torch.cat(all_offsets)
        all_offsets = torch.cat([torch.Tensor([0]), all_offsets]).to(torch.int64)
        all_offsets = torch.cumsum(all_offsets, dim=0)

        all_idx = all_idx.to(DEVICEID)
        all_offsets = all_offsets.to(DEVICEID)

        weights_no_grad_aggregation = look_table(all_idx, all_offsets, all_jt_lst, tbe_no_grad_aggregation, params)

    verify = torch.allclose(weights_grad_aggregation, weights_no_grad_aggregation, 1e-4, 1e-4)
    print('weights_test', weights_test)
    print('weights_grad_aggregation', weights_grad_aggregation)
    print('weights_no_grad_aggregation', weights_no_grad_aggregation)
    print('verify', torch.eq(weights_grad_aggregation, weights_no_grad_aggregation))
    print('allclose', verify)

    raise ValueError(f"verify grad aggregation is {verify}")


@pytest.mark.parametrize("tables", [[(10, 8), (5, 8), (4, 8)]])
@pytest.mark.parametrize("mutile_hots", [[1, 1, 1]])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("unique", [True])
@pytest.mark.parametrize("feature_map", [[0, 1, 2]])
@pytest.mark.parametrize("pooling_model", [PoolingType.NONE])
@pytest.mark.parametrize("optim", [Adagrad])
def test_verify_grad_aggregation(tables, mutile_hots, batch_size, pooling_model, unique, optim, feature_map):
    params = LookupParams(tables, mutile_hots, batch_size, pooling_model, unique, optim, feature_map)
    verify_grad_aggregation(params)