#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

import sysconfig
import logging
import random
from collections import defaultdict

import pytest
import torch
import torchrec
import torch_npu
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    SplitTableBatchedEmbeddingBagsCodegen,
    ComputeDevice,
)
from hybrid_torchrec.distributed.batched_embedding_kernel import HybridSplitTableBatchedEmbeddingBagsCodegen
from torch.optim import Adam, Adagrad, SGD
from torchrec import JaggedTensor, KeyedJaggedTensor, PoolingType

logging.getLogger().setLevel(logging.INFO)
device_id = "npu:0"
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


class TestModel(torch.nn.Module):
    def __init__(self, weights, tables, mode):
        super().__init__()
        if mode == PoolingType.NONE:
            collection = torchrec.EmbeddingCollection
            param_name = "embeddings"
        else:
            collection = torchrec.EmbeddingBagCollection
            param_name = "embedding_bags"

        self.param_name = param_name
        self.table_names = list(map(lambda x: x.name, tables))

        self.ec = collection(device="cpu", tables=tables)
        for (table, weight) in zip(tables, weights):
            self.get_table_weights(table.name).copy_(weight)

    def forward(self, kjt):
        return self.ec(kjt)

    def get_table_weights(self, table_name):
        table_dict = getattr(self.ec, self.param_name)
        return table_dict[table_name].weight.data

    def get_all_tables_weights(self):
        return list(map(self.get_table_weights, self.table_names))


def construct_collection_configs(weights, tables, mode, feature_map):
    if mode == PoolingType.NONE:
        table_config_type = torchrec.EmbeddingConfig
        pooling_mode_dict = dict()
    else:
        table_config_type = torchrec.EmbeddingBagConfig
        pooling_mode_dict = dict(pooling=mode)

    features = defaultdict(list)
    for ind, tid in enumerate(feature_map):
        features[f"t_{tid}"].append(f"f_{ind}")

    table_configs_list, table_weights_list = [], []
    weights_offset = 0
    for table_id, (num_embeddings, embedding_dim) in enumerate(tables):
        table_name = f"t_{table_id}"
        table_configs = table_config_type(
            name=table_name,
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            feature_names=features[table_name],
            **pooling_mode_dict
        )

        # 将一维的weights整理成多张二位的embedding表
        table_size = num_embeddings * embedding_dim
        table_weights = weights[weights_offset:weights_offset + table_size]
        table_weights = table_weights.reshape(num_embeddings, embedding_dim)

        table_configs_list.append(table_configs)
        table_weights_list.append(table_weights)

        weights_offset += table_size
    return table_weights_list, table_configs_list


def lookup_cpu(kjt, weights, tables, mode, optim, feature_map):
    collection_configs = construct_collection_configs(weights, tables, mode, feature_map)
    model = TestModel(*collection_configs, mode)
    model.zero_grad()
    optimizer = optim(model.parameters(), **OPTIMIZER_PARAM[optim])

    output = None
    for _ in range(EPOCH):
        # forward
        output = model(kjt)

        # 将多个表的查询结果合并
        reshaped_output = [jt.values() if isinstance(jt, JaggedTensor) else jt.view(1, -1) for jt in output.values()]
        output = torch.cat(reshaped_output, dim=0)
    updated_weights = list(map(lambda x: x.reshape(-1), model.get_all_tables_weights()))
    updated_weights = torch.cat(updated_weights, dim=0)
    return output, updated_weights


def lookup_npu(indices, offsets, weights, jt_lst, pooling_mode, tables, optim, unique, feature_map):
    torch.npu.set_device(device_id)

    indices = indices.to(device_id)
    offsets = offsets.to(device_id)
    weights = weights.to(device_id)

    embedding_specs = [
        (num_embeddings, embedding_dim, EmbeddingLocation.DEVICE, ComputeDevice.NPU)
        for (num_embeddings, embedding_dim) in tables
    ]
    if unique:
        ebc_class = HybridSplitTableBatchedEmbeddingBagsCodegen
        unique_indices, unique_inverse, unique_offset = generate_unique(jt_lst, feature_map)
        unique_indices = torch.cat(unique_indices).to(device_id).to(torch.int64)
        unique_inverse = torch.cat(unique_inverse).to(device_id).to(torch.int64)
        unique_offset = torch.Tensor(unique_offset).to(device_id).to(torch.int64)
        kwargs = dict(unique_indices=unique_indices, unique_offset=unique_offset, unique_inverse=unique_inverse)
    else:
        ebc_class = SplitTableBatchedEmbeddingBagsCodegen
        kwargs = dict()

    tbe = ebc_class(
        embedding_specs,
        optimizer=TORCH_OPTIMIZER_TO_FBGEMM[optim],
        device=torch.device(device_id),
        pooling_mode=TORCH_POOLING_MODE_TO_FBGEMM[pooling_mode],
        feature_table_map=feature_map,
    )

    tbe.weights_dev = torch.nn.Parameter(weights.clone()).to(device_id)

    output = tbe(indices, offsets, **kwargs)
    return output, tbe.weights_dev


def create_data(tables, mutile_hots, batch_size, feature_map):
    total_size = sum([num_embeddings * embedding_dim for (num_embeddings, embedding_dim) in tables])

    indices_test = []
    offsets_test = []
    jt_lst = []
    for ind, tid in enumerate(feature_map):
        table = tables[tid]
        indices = torch.randint(0, table[0], (batch_size * mutile_hots[ind],)).to(torch.int64)
        indices_test.append(indices)
        offsets = torch.Tensor([mutile_hots[ind] for _ in range(batch_size)]).to(torch.int64)
        offsets_test.append(offsets)

        jt_lst.append(JaggedTensor(values=indices, lengths=offsets))

    indices_test = torch.cat(indices_test).to(torch.int64)
    offsets_test = torch.cat(offsets_test).to(torch.int64)
    offsets_test = torch.cat([torch.Tensor([0]), offsets_test]).to(torch.int64)
    offsets_test = torch.cumsum(offsets_test, dim=0)

    weights_test = torch.randn(total_size).to(torch.float32)

    jt_dict = {f"f_{i}": jt for i, jt in enumerate(jt_lst)}
    kjt = KeyedJaggedTensor.from_jt_dict(jt_dict)

    return indices_test, offsets_test, weights_test, kjt, jt_lst


def generate_tables(pooling_model, tb_num=10, bs=100, rows=20000, dims=100, offset=100):
    tables = []
    mutile_hots = []
    batches = random.randint(1, bs)
    table_num = random.randint(1, tb_num)
    embed_dim = random.randint(1, dims) * 8
    for _ in range(table_num):
        row = random.randint(1, rows)
        if pooling_model == PoolingType.NONE:
            col = embed_dim
        else:
            col = random.randint(1, dims) * 8
        tables.append((row, col))
        mutile_hots.append(random.randint(1, offset))
    return tables, mutile_hots, batches


def generate_unique(jt_lst, feature_map):
    unique_indices = []
    unique_inverse = []
    unique_offset = []
    start = 0
    # 合并同一个表的不同feature
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


def execute(tables, mutile_hots, batch_size, pooling_model, unique, optim, feature_map=None):
    if unique and (optim == SGD):
        return  # 暂未适配adam unique算子
    if feature_map is None:
        feature_map = list(range(len(tables)))
    indices_test, offsets_test, weights_test, kjt, jt_lst = create_data(tables, mutile_hots, batch_size, feature_map)

    lookup_golden, weights_golden = lookup_cpu(kjt, weights_test, tables, pooling_model, optim, feature_map)
    lookup_npu_result, weights_npu_result = lookup_npu(indices_test, offsets_test, weights_test, jt_lst,
                                                       pooling_model, tables, optim, unique, feature_map)

    total_size = sum([num_embeddings * embedding_dim for (num_embeddings, embedding_dim) in tables])
    lookup_npu_result = lookup_npu_result.detach().cpu()
    weights_npu_result = weights_npu_result.detach().cpu()

    logging.info("====== forward ===========")
    lookup_compare = torch.isclose(lookup_golden, lookup_npu_result, 1e-4, 1e-4)
    logging.info((~lookup_compare).sum())
    logging.info(lookup_npu_result[~lookup_compare])
    logging.info(lookup_golden[~lookup_compare])

    assert (~lookup_compare).sum() == 0


@pytest.mark.parametrize("tables", [[(20000, 32), (40000, 32)], [(40000, 128), (80000, 128)]])
@pytest.mark.parametrize("mutile_hots", [[8, 16, 100], [2, 64, 200]])
@pytest.mark.parametrize("batch_size", [8, 16, 64])
@pytest.mark.parametrize("unique", [True, False])
@pytest.mark.parametrize("feature_map", [[0, 0, 1], [0, 1, 1]])
@pytest.mark.parametrize("pooling_model", [PoolingType.SUM, PoolingType.MEAN, PoolingType.NONE])
@pytest.mark.parametrize("optim", [Adam, Adagrad, SGD])
def test_lookup_two_tables(tables, mutile_hots, batch_size, pooling_model, unique, optim, feature_map):
    execute(tables, mutile_hots, batch_size, pooling_model, unique, optim, feature_map)


@pytest.mark.parametrize("tables", [[(10240, 1024)], [(1234, 1536)], [(1, 8)]])
@pytest.mark.parametrize("mutile_hots", [[1], [4], [11], [69]])
@pytest.mark.parametrize("batch_size", [2341, 1])
@pytest.mark.parametrize("unique", [True, False])
@pytest.mark.parametrize("pooling_model", [PoolingType.SUM, PoolingType.MEAN, PoolingType.NONE])
@pytest.mark.parametrize("optim", [Adam, Adagrad, SGD])
def test_lookup_backward_one_table(tables, mutile_hots, batch_size, pooling_model, unique, optim):
    execute(tables, mutile_hots, batch_size, pooling_model, unique, optim)


@pytest.mark.parametrize("unique", [True, False])
@pytest.mark.parametrize("pooling_model", [PoolingType.SUM, PoolingType.MEAN, PoolingType.NONE])
@pytest.mark.parametrize("optim", [Adam, Adagrad, SGD])
def test_lookup_multi_tables(pooling_model, unique, optim):
    tables, mutile_hots, batches = generate_tables(pooling_model)
    execute(tables, mutile_hots, batches, pooling_model, unique, optim)
