#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import random
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor

import pytest
import torchrec
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.optim import Adam, Adagrad

from dataset import (
    RandomRecDataset, 
    Batch, 
    BoundOutOfRangeRecDataset, 
    FeatureNameNotInConfigRecDataset
)
from dt.conftest import MODULE_NAME
from model import TestModel, generate_hash_config
from torchrec import EmbeddingBagConfig, EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec_embcache.distributed.train_pipeline import (
    AwaitableAdapter,
    EmbcacheTrainPipelineContext,
)
from util import (
    setup_logging,
    is_lookup_out_of_bound,
    feature_name_exists,
    create_weight_init,
    check_config,
    TEST_ROOT_DIR,
    OVER_COUNT,
    compare_tensors,
    compare_lists,
    fuse_input_dist_splits,
)


@pytest.mark.functional
def test_normal(request, config):
    fname = request.node.callspec.id
    config["fname"] = fname
    run_model_with_config(config)


@pytest.mark.functional
def test_table_num_invalid(request, config):
    fname = request.node.callspec.id
    config["fname"] = fname
    assert config["table_num"] < 1
    with ProcessPoolExecutor() as executor:
        future = executor.submit(run_model_with_config, config)
        with pytest.raises(Exception) as exc_info:
            future.result()

    assert "ValueError" in str(exc_info.value)


@pytest.mark.functional
def test_embedding_dim_invalid(request, config):
    fname = request.node.callspec.id
    config["fname"] = fname
    assert any([embedding_dim < 1 or embedding_dim % 4 for embedding_dim in config["embedding_dims"]])
    with ProcessPoolExecutor() as executor:
        future = executor.submit(run_model_with_config, config)
        with pytest.raises(Exception) as exc_info:
            future.result()

    assert "ValueError" in str(exc_info.value)


@pytest.mark.functional
def test_num_embeddings_invalid(request, config):
    fname = request.node.callspec.id
    config["fname"] = fname
    assert any([num_embedding < 1 for num_embedding in config["num_embeddings"]])
    with ProcessPoolExecutor() as executor:
        future = executor.submit(run_model_with_config, config)
        with pytest.raises(Exception) as exc_info:
            future.result()

    assert "ValueError" in str(exc_info.value)


# HBM需要 DDR不需要
@pytest.mark.functional
def test_lookup_out_of_bound(request, config):
    fname = request.node.callspec.id
    config["fname"] = fname
    assert is_lookup_out_of_bound(config)
    with ProcessPoolExecutor() as executor:
        future = executor.submit(run_model_with_config, config)
        with pytest.raises(Exception) as exc_info:
            future.result()

    assert "IndexError" in str(exc_info.value)


@pytest.mark.functional
def test_feature_name_exist(request, config):
    fname = request.node.callspec.id
    config["fname"] = fname
    assert feature_name_exists(config)
    with ProcessPoolExecutor() as executor:
        future = executor.submit(run_model_with_config, config)
        with pytest.raises(Exception) as exc_info:
            future.result()

    assert "KeyError" in str(exc_info.value)


def run_model_with_config(config):
    if config.get("device", "npu") == "cpu" and config.get("sharding_type", "table_wise") == "row_wise":
        return
    mp.spawn(
        execute,
        args=(config,),
        nprocs=config.get("WORLD_SIZE", 2),
        join=True,
    )


def are_features_equal(obj1, obj2):
    attributes_to_compare = ["embs", "optims"]

    for attr in attributes_to_compare:
        value1 = getattr(obj1, attr, None)
        value2 = getattr(obj2, attr, None)

        if value1 is None or value2 is None:
            logging.error(f"Attribute '{attr}' not found in one of the objects.")
            return False
        elif isinstance(value1, list):
            if not compare_lists(value1, value2):
                logging.debug("Lists are not equal: %s != %s", value1, value2)
                return False
        elif isinstance(value1, torch.Tensor):
            if not compare_tensors(value1, value2):
                logging.debug("Tensors are not equal: %s != %s", value1, value2)
                return False
    return True


def execute(rank, config):
    setup_logging(rank)
    logging.info("this test %s", os.path.basename(__file__))
    check_config(config)
    embedding_dims = config["embedding_dims"]
    num_embeddings = config["num_embeddings"]
    pool_type = config["pool_type"]
    BATCH_NUM = config["BATCH_NUM"]
    table_num = config["table_num"]
    lookup_lens = config["lookup_lens"]
    dataset_class = globals()[config["RecDataset"] + "RecDataset"]
    init_fn = globals()[config["init_fn"]]
    WORLD_SIZE = config["WORLD_SIZE"]
    device = config.get("device", "npu")
    sharding_type = config.get("sharding_type", "row_wise")
    optim = globals()[config.get("optim", "Adagrad")]
    feature_names_lst = config["feature_names_lst"]
    instances = config.get("instances", 1)
    pool_type = getattr(torchrec.PoolingType, pool_type)
    collection_type = config["collection_type"]
    embedding_config = generate_hash_config(embedding_dims, num_embeddings, pool_type, feature_names_lst, 
                                            create_weight_init(init_fn), collection_type)
    generated_ids = []
    if isinstance(dataset_class, BoundOutOfRangeRecDataset):
        for i in range(table_num):
            generated_ids.append([])
            for _ in range(len(feature_names_lst[i])):
                generated_ids[i].append(list(range(num_embeddings[i]+OVER_COUNT)))
                random.shuffle(generated_ids[i][-1])
    dataset = dataset_class(BATCH_NUM, lookup_lens, num_embeddings, table_num, feature_names_lst, generated_ids)
    data_loader = DataLoader(
        dataset,
        batch_size=None,
        pin_memory=True,
        pin_memory_device="npu",
        num_workers=1,
    )

    test_model = TestModel(rank, WORLD_SIZE, device, instances, feature_names_lst, BATCH_NUM, collection_type)
    test_model.init_ddp_model(embedding_config, sharding_type, optim, lookup_lens)
    iter_ = iter(data_loader)
    module_lst = getattr(test_model.module, collection_type)
    context = EmbcacheTrainPipelineContext(index=0, version=1)

    for i, module in enumerate(module_lst):
        name = f"module.{i}"
        ctx = module.create_context()
        features = next(iter_).sparse_features
        context.input_dist_splits_requests[name] = module.input_dist(ctx, features)
        context.module_contexts[name] = ctx

    fuse_input_dist_splits(context)

    for names, awaitable in context.fused_splits_awaitables:
        for name, request in zip(names, awaitable.wait()):
            context.input_dist_tensors_requests[name] = AwaitableAdapter(request)

    swapout_tensor_dict_lst = []
    swapout_dict_lst = []
    for i, module in enumerate(module_lst):
        name = f"module.{i}"
        awaitable = context.input_dist_tensors_requests[name]
        kjt_list = awaitable.wait()
        post_waitable = module.post_input_dist(
            context.module_contexts[name],
            kjt_list,
        )
        sparse_features = post_waitable.wait()
        swap_info_future = module.compute_swap_info_async(sparse_features)
        swap_info = swap_info_future.wait()
        sparse_features[0]._unique_indices = swap_info.batch_offs
        for j in range(len(sparse_features)):
            sparse_features[j] = sparse_features[j].to(test_model.npu_device, non_blocking=True)
        swap_info.swapout_keys = swap_info.swapin_keys
        swap_info.swapout_offs = swap_info.swapin_offs.to(test_model.npu_device, non_blocking=True)
        swap_info.swapin_offs = swap_info.swapin_offs.to(test_model.npu_device, non_blocking=True)

        _stb_eb_codegen = module.get_batched_embedding_kernels()[0][0]

        swap_offs = swap_info.swapout_offs
        swapout_embs = _stb_eb_codegen.gather_embs(swap_offs).to(test_model.npu_device, non_blocking=True)
        swapout_optims = []
        for momentum in _stb_eb_codegen.get_momentum(swap_offs):
            swapout_optims.append(momentum.to(test_model.npu_device, non_blocking=True))
        # 未知原因，需要logging.debug或者print才能顺利用lookup进行查询，否则为全0
        logging.debug("swapout_embs: %s", swapout_embs)
        logging.debug("swapout_optims: %s", swapout_optims)
        update_future = module.host_embedding_update_async(swap_info, swapout_embs, swapout_optims)
        update_future.get()
        swapout_tensor_future = module.host_embedding_lookup_async(swap_info)
        swapout_tensor = swapout_tensor_future.get()
        swapout_dict = {
            "embs": swapout_embs,
            "optims": swapout_optims,
        }
        swapout_tensor_dict = {
            "embs": swapout_tensor.swapin_embs,
            "optims": swapout_tensor.swapin_optims,
        }
        swapout_dict_lst.append(swapout_dict)
        swapout_tensor_dict_lst.append(swapout_tensor_dict)

    if not config["fname"].startswith("test_normal"):
        logging.debug("Skipping saving baseline for non-normal test: %s", config["fname"])
        return

    for obj1, obj2 in zip(swapout_tensor_dict_lst, swapout_dict_lst):
        assert are_features_equal(obj1, obj2), "swapout_tensor_dict and swapout_dict are not equal"