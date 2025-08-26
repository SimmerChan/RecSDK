#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
import itertools
import logging
import os
import shutil
from typing import List

import numpy as np
import pytest
import torch
import torch_npu
import torch.multiprocessing as mp
import torch.distributed as dist
from torch import nn, Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchrec_embcache.distributed.embedding import EmbCacheEmbeddingCollection
from torchrec_embcache.distributed.configs import (EmbCacheEmbeddingConfig,
                                                   AdmitAndEvictConfig)
from torchrec_embcache.distributed.train_pipeline import EmbCacheTrainPipelineSparseDist
from torchrec_embcache.distributed.sharding.embedding_sharder import EmbCacheEmbeddingCollectionSharder
from torchrec_embcache.sparse.jagged_tensor_with_timestamp import KeyedJaggedTensorWithTimestamp
import torchrec
import torchrec.distributed
from torchrec import EmbeddingCollection
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    Topology,
    ParameterConstraints,
)
from torchrec.distributed.types import ShardingEnv
from torchrec.optim.keyed import CombinedOptimizer

from dataset import RandomRecDataset, Batch
from model import ModelEc as Model
from util import setup_logging

_SAVE_PATH = "save_dir/sparse"

WORLD_SIZE_STR = str(os.environ.get("WORLD_SIZE", "2"))
WORLD_SIZE = int(WORLD_SIZE_STR) if WORLD_SIZE_STR.isalnum() else 2
LOOP_TIMES = 500
EVICT_STEP_INTERVAL = LOOP_TIMES // 4
BATCH_NUM = LOOP_TIMES


def _check_admit_key_count(data_loader_golden, embedding_configs: List[EmbCacheEmbeddingConfig], rank):
    # 1 手动统计key count
    iter_ = iter(data_loader_golden)
    loop_time = 0
    table_key_count = [{} for _ in range(len(embedding_configs))]

    while loop_time < LOOP_TIMES:
        loop_time += 1
        batch: Batch = next(iter_, None)
        if batch is None:
            break
        kjt = batch.sparse_features
        if len(kjt.keys()) != len(embedding_configs):
            raise ValueError("key num should equal with embedding_configs length")
        values = kjt.values()
        offset_per_key = kjt.offset_per_key()
        for i in range(len(offset_per_key) - 1):
            values_per_table = values[offset_per_key[i]: offset_per_key[i + 1]]
            for ids in values_per_table:
                ids = ids.item()
                if ids % WORLD_SIZE != rank:
                    continue
                if ids in table_key_count[i]:
                    table_key_count[i][ids] = table_key_count[i][ids] + 1
                else:
                    table_key_count[i][ids] = 1

    # 2 读取保存目录下的key count
    key_file_saved = os.path.join(_SAVE_PATH, "table{}", "rank{}".format(rank), "key", "slice.data")
    count_file_saved = os.path.join(_SAVE_PATH, "table{}", "rank{}".format(rank), "admit_count", "slice.data")
    table_key_count_saved = [{} for _ in range(len(embedding_configs))]
    for i in range(len(embedding_configs)):
        if not os.path.exists(key_file_saved.format(i)):
            raise ValueError(f"file:{key_file_saved.format(i)} is not exist when check key count data.")
        if not os.path.exists(count_file_saved.format(i)):
            raise ValueError(f"file:{count_file_saved.format(i)} is not exist when check key count data.")
        key_data = np.fromfile(key_file_saved.format(i), dtype=np.int64).reshape(-1)
        count_data = np.fromfile(count_file_saved.format(i), dtype=np.int64).reshape(-1)
        for index in range(key_data.shape[0]):
            ids = key_data[index]
            count = count_data[index]
            table_key_count_saved[i][ids] = count

    logging.info("rankId:" + str(rank) + ", table_key_count:%s", table_key_count)
    logging.info("rankId:" + str(rank) + ", table_key_count_saved:%s", table_key_count_saved)

    # 3 对比数据
    length_equal = all(len(table_key_count[i]) == len(table_key_count_saved[i]) for i in range(len(embedding_configs)))
    assert length_equal, "key count length is not equal."
    for i in range(len(embedding_configs)):
        keys = table_key_count[i].keys()
        for key in keys:
            # 手动统计key count * WORLD_SIZE 即为所有卡all2all通信后key count
            count_equal = table_key_count[i][key] * WORLD_SIZE == table_key_count_saved[i][key]
            assert count_equal, "key count value is not equal."


@dataclass
class ExecuteConfig:
    world_size: int
    table_num: int
    embedding_dims: List[int]
    num_embeddings: List[int]
    sharding_type: str
    lookup_len: int
    device: str
    enable_admit: bool
    enable_evict: bool


def execute(rank: int, config: ExecuteConfig):
    world_size = config.world_size
    table_num = config.table_num
    embedding_dims = config.embedding_dims
    num_embeddings = config.num_embeddings
    sharding_type = config.sharding_type
    lookup_len = config.lookup_len
    device = config.device
    enable_admit = config.enable_admit
    enable_evict = config.enable_evict
    setup_logging(rank)
    logging.info("this test %s", os.path.basename(__file__))

    dataset = RandomRecDataset(BATCH_NUM, lookup_len, num_embeddings, table_num, is_evict_enabled=enable_evict)
    dataset_golden = RandomRecDataset(BATCH_NUM, lookup_len, num_embeddings, table_num, is_evict_enabled=enable_evict)
    data_loader_golden = DataLoader(
        dataset_golden,
        batch_size=None,
        batch_sampler=None,
        pin_memory=True,
        pin_memory_device="npu",
        num_workers=1,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=None,
        batch_sampler=None,
        pin_memory=True,
        pin_memory_device="npu",
        num_workers=1,
    )
    embedding_configs = []
    default_config = AdmitAndEvictConfig()
    admit_threshold = 2 if enable_admit else default_config.admit_threshold
    evict_threshold = 2000_0000 if enable_evict else default_config.evict_threshold
    for i in range(table_num):
        admit_and_evict_config = AdmitAndEvictConfig(admit_threshold=admit_threshold,
                                                     not_admitted_default_value=0.999,
                                                     evict_threshold=evict_threshold,
                                                     evict_step_interval=EVICT_STEP_INTERVAL)
        ec_config = EmbCacheEmbeddingConfig(
            name=f"table{i}",
            embedding_dim=embedding_dims[i],
            num_embeddings=num_embeddings[i],
            feature_names=[f"feat{i}"],
            init_fn=weight_init,
            weight_init_min=0.0,
            weight_init_max=1.0,
            admit_and_evict_config=admit_and_evict_config
        )
        embedding_configs.append(ec_config)

    test_model = TestModel(rank, world_size, device)
    test_result_golden = []
    if not enable_admit and enable_evict:
        test_result_golden = test_model.cpu_golden_loss(embedding_configs, data_loader_golden, evict_threshold, rank)
    test_results = test_model.test_loss(embedding_configs, data_loader, sharding_type, enable_evict, training=True)

    # load
    test_model.test_loss(embedding_configs, data_loader_golden, sharding_type, enable_evict, training=False)

    for i, result in enumerate(test_results):
        logging.debug("")
        logging.debug("==============batch %d================", i // 2)
        logging.debug("result test %s", result)
        # check evict ret
        if not enable_admit and enable_evict:
            golden = test_result_golden[i]
            logging.debug("golden test %s", golden)
            assert torch.allclose(
                golden, result, rtol=1e-04, atol=1e-04
            ), "golden and result is not closed"
    dist.destroy_process_group()


def weight_init(param: torch.nn.Parameter):
    if len(param.shape) != 2:
        return
    torch.manual_seed(param.shape[1])
    result = torch.linspace(0, 1, steps=param.shape[1]).unsqueeze(0).repeat(param.shape[0], 1)
    param.data.copy_(result)


def _get_init_weight(table_dims: List[int]):
    init_embs = []
    for dim in table_dims:
        emb = torch.linspace(0, 1, steps=dim)
        init_embs.append(emb)
    return init_embs


def _get_init_optimizer_slot(table_dims: List[int]):
    init_slots = []
    for dim in table_dims:
        slot = torch.zeros((dim,))
        init_slots.append(slot)
    return init_slots


class TestModel:
    def __init__(self, rank, world_size, device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.pg_method = "hccl" if device == "npu" else "gloo"
        if device == "npu":
            torch_npu.npu.set_device(rank)
        self.setup(rank=rank, world_size=world_size)
        self.emb_configs: List[EmbCacheEmbeddingConfig] = []

        # for evict 
        self.timestamps_for_table: List[dict] = []
        self.last_timestamp_for_table = []

    def setup(self, rank: int, world_size: int):
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "6015"
        dist.init_process_group(self.pg_method, rank=rank, world_size=world_size)
        os.environ["LOCAL_RANK"] = f"{rank}"

    def test_loss(
            self,
            embedding_configs: List[EmbCacheEmbeddingConfig],
            dataloader: DataLoader[Batch],
            sharding_type: str,
            enable_evict: bool,
        training: bool = True,
        ):
        rank, world_size = self.rank, self.world_size
        host_gp = dist.new_group(backend="gloo")
        host_env = ShardingEnv(world_size=world_size, rank=rank, pg=host_gp)

        table_num = len(embedding_configs)
        ec = EmbCacheEmbeddingCollection(device=torch.device("meta"), tables=embedding_configs,
                                         batch_size=2, multi_hot_sizes=[1] * table_num,
                                         world_size=dist.get_world_size())
        num_features = sum([c.num_features() for c in embedding_configs])
        ec = Model(ec, num_features)
        apply_optimizer_in_backward(
            optimizer_class=torch.optim.Adagrad,
            params=ec.parameters(),
            optimizer_kwargs={"lr": 0.02},
        )
        # Shard
        constrains = {
            f"table{i}": ParameterConstraints(sharding_types=[sharding_type])
            for i in range(table_num)
        }
        rank = int(os.environ["LOCAL_RANK"])
        npu_device: torch.device = torch.device(f"npu:{rank}")
        cpu_device = torch.device("cpu")
        cpu_pg = dist.new_group(backend="gloo")
        cpu_env = ShardingEnv.from_process_group(cpu_pg)
        hash_shader = EmbCacheEmbeddingCollectionSharder(
            cpu_device=cpu_device,
            cpu_env=cpu_env,
            npu_device=npu_device,
            npu_env=ShardingEnv.from_process_group(dist.GroupMember.WORLD),
        )
        shaders = [hash_shader]
        planner = EmbeddingShardingPlanner(
            topology=Topology(world_size=self.world_size, compute_device=self.device),
            constraints=constrains,
        )
        plan = planner.collective_plan(
            ec, shaders, dist.GroupMember.WORLD
        )
        if self.rank == 0:
            logging.debug(plan)

        ddp_model = torchrec.distributed.DistributedModelParallel(
            ec,
            sharders=shaders,
            device=npu_device,
            plan=plan,
        )

        logging.debug(ddp_model)
        # Optimizer
        optimizer = CombinedOptimizer([ddp_model.fused_optimizer])
        results = []
        if training:
            iter_ = iter(dataloader)
            ddp_model.train()
            evict_step_interval = EVICT_STEP_INTERVAL if enable_evict else None
            pipe = EmbCacheTrainPipelineSparseDist(
                ddp_model,
                optimizer=optimizer,
                cpu_device=cpu_device,
                npu_device=npu_device,
                return_loss=True,
                evict_step_interval=evict_step_interval
            )

            for _ in range(LOOP_TIMES):
                out, loss = pipe.progress(iter_)
                results.append(loss.detach().cpu())
                results.append(out.detach().cpu())

            save_dir = os.path.abspath("save_dir")
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir, ignore_errors=True)
            os.makedirs(save_dir, exist_ok=True) 

        return results

    def cpu_golden_loss(self, embedding_configs: List[EmbCacheEmbeddingConfig], dataloader: DataLoader[Batch],
                        evict_threshold: int, rank_id: int):
        pg = dist.new_group(backend="gloo")
        self.emb_configs = embedding_configs
        table_num = len(embedding_configs)
        ec = EmbeddingCollection(device=torch.device("cpu"), tables=embedding_configs)

        num_features = sum([c.num_features() for c in embedding_configs])
        ec_wrap = Model(ec, num_features)
        model = DDP(ec_wrap, process_group=pg)

        opt = torch.optim.Adagrad(model.parameters(), lr=0.02, eps=1e-8)
        results = []
        batch: Batch
        iter_ = iter(dataloader)
        for i in range(LOOP_TIMES):
            batch = next(iter_)
            opt.zero_grad()
            loss, outputs = model(batch)
            results.append(loss.detach().cpu())
            results.append(outputs.detach().cpu())
            loss.backward()
            opt.step()

            # 1 record batch timestamp data
            self._record_timestamp_info_cpu(batch, table_num, i)
            # 2 evict emb and optimizer data
            if i > 0 and (i + 1) % EVICT_STEP_INTERVAL == 0:
                self._evict_embedding_cpu(evict_threshold, ec.embeddings, opt, i)

        return results

    def _record_timestamp_info_cpu(self, batch, table_num, batch_id):
        sparse_tensor: KeyedJaggedTensorWithTimestamp = batch.sparse_features
        values = sparse_tensor.values()
        timestamps = sparse_tensor.timestamps
        offset_per_key = sparse_tensor.offset_per_key()
        # init data structure
        if len(self.timestamps_for_table) == 0:
            for _ in range(table_num):
                self.timestamps_for_table.append(dict())
                self.last_timestamp_for_table.append(0)
        
        # record timestamp data
        for table_index in range(table_num):
            start = offset_per_key[table_index]
            end = offset_per_key[table_index + 1]
            values_per_table = values[start:end]
            ts_per_table = timestamps[start:end]

            for index, ids in enumerate(values_per_table):
                ids = ids.item()
                ts = ts_per_table[index].item()
                self.timestamps_for_table[table_index][ids] = ts
                self.last_timestamp_for_table[table_index] = max(self.last_timestamp_for_table[table_index], ts)

    def _evict_embedding_cpu(self, evict_threshold: int, embeddings: nn.ModuleDict,
                             opt: torch.optim.Adagrad, batch_id: int):
        logging.info("Start cpu embedding evict, current step:%d", batch_id)
        emb_dims: List[int] = [c.embedding_dim for c in self.emb_configs]
        table_names = [c.name for c in self.emb_configs]
        table_num = len(table_names)
        emb_init_values: List[Tensor] = _get_init_weight(emb_dims)
        optimizer_init_values: List[Tensor] = _get_init_optimizer_slot(emb_dims)
        for table_index in range(table_num):
            evict_ids_per_table = []
            last_timestamp = self.last_timestamp_for_table[table_index]
            for ids, ts in self.timestamps_for_table[table_index].items():
                if last_timestamp - ts > evict_threshold:
                    evict_ids_per_table.append(ids)

            table_name = table_names[table_index]
            # get slot tensor of Adagrad optimizer
            op_t = opt.param_groups[0]["params"][table_index]
            slot_tensor = opt.state[op_t]["sum"]
            for ids in evict_ids_per_table:
                # step1 delete timestamp record for ids
                self.timestamps_for_table[table_index].pop(ids)
                # step2 reset emb and optimizer slot as init value
                with torch.no_grad():
                    # init emb
                    embeddings[table_name].weight[ids].data.copy_(emb_init_values[table_index])
                    # init optimizer slot
                    slot_tensor[ids].data.copy_(optimizer_init_values[table_index])
            logging.info("batchId:%d, table name:%s, evict ids num:%d",
                         batch_id, table_name, len(evict_ids_per_table))
    

params = {
    "world_size": [WORLD_SIZE],
    "table_num": [2],
    "embedding_dims": [[128, 128]],
    "num_embeddings": [[4000, 400]],
    "sharding_type": ["row_wise"],
    "lookup_len": [128],  # batchsize
    "device": ["npu"],
    "enable_admit": [True],
    "enable_evict": [True],
}


@pytest.mark.parametrize("config", [
    ExecuteConfig(*v) for v in itertools.product(*params.values())
])
def test_hstu_dens_normal(config: ExecuteConfig):
    mp.spawn(
        execute,
        args=(config,),
        nprocs=WORLD_SIZE,
        join=True,
    )


params = {
    "world_size": [WORLD_SIZE],
    "table_num": [2],
    "embedding_dims": [[128, 128]],
    "num_embeddings": [[4000, 400]],
    "sharding_type": ["row_wise"],
    "lookup_len": [128],  # batchsize
    "device": ["npu"],
    "enable_admit": [True],
    "enable_evict": [False],
}


@pytest.mark.parametrize("config", [
    ExecuteConfig(*v) for v in itertools.product(*params.values())
])
def test_admit_count_correctness(config: ExecuteConfig):
    mp.spawn(
        execute,
        args=(config,),
        nprocs=WORLD_SIZE,
        join=True,
    )


params = {
    "world_size": [WORLD_SIZE],
    "table_num": [2],
    "embedding_dims": [[128, 128]],
    "num_embeddings": [[4000, 400]],
    "sharding_type": ["row_wise"],
    "lookup_len": [128],  # batchsize
    "device": ["npu"],
    "enable_admit": [False],
    "enable_evict": [True],
}


@pytest.mark.parametrize("config", [
    ExecuteConfig(*v) for v in itertools.product(*params.values())
])
def test_evict_correctness(config: ExecuteConfig):
    mp.spawn(
        execute,
        args=(config,),
        nprocs=WORLD_SIZE,
        join=True,
    )


# 测试tableToFilterIndexMap_相关功能
params = {
    "world_size": [WORLD_SIZE],
    "table_num": [3],  # 测试多个表
    "embedding_dims": [[64, 128, 256]],
    "num_embeddings": [[1000, 2000, 3000]],
    "sharding_type": ["row_wise"],
    "lookup_len": [64],
    "device": ["npu"],
    "enable_admit": [True],  # 第1、3个表开启准入
    "enable_evict": [False],
}


@pytest.mark.parametrize("config", [
    ExecuteConfig(*v) for v in itertools.product(*params.values())
])
def test_table_to_filter_index_mapping_admit_only(config: ExecuteConfig):
    """测试只开启准入时的tableToFilterIndexMap映射关系"""
    mp.spawn(
        execute_with_custom_filter_config,
        args=(config, "admit_only"),
        nprocs=WORLD_SIZE,
        join=True,
    )


params = {
    "world_size": [WORLD_SIZE],
    "table_num": [3],
    "embedding_dims": [[64, 128, 256]],
    "num_embeddings": [[1000, 2000, 3000]],
    "sharding_type": ["row_wise"],
    "lookup_len": [64],
    "device": ["npu"],
    "enable_admit": [False],
    "enable_evict": [True],  # 第2、3个表开启淘汰
}


@pytest.mark.parametrize("config", [
    ExecuteConfig(*v) for v in itertools.product(*params.values())
])
def test_table_to_filter_index_mapping_evict_only(config: ExecuteConfig):
    """测试只开启淘汰时的tableToFilterIndexMap映射关系"""
    mp.spawn(
        execute_with_custom_filter_config,
        args=(config, "evict_only"),
        nprocs=WORLD_SIZE,
        join=True,
    )


params = {
    "world_size": [WORLD_SIZE],
    "table_num": [4],
    "embedding_dims": [[32, 64, 128, 256]],
    "num_embeddings": [[800, 1600, 3200, 6400]],
    "sharding_type": ["row_wise"],
    "lookup_len": [32],
    "device": ["npu"],
    "enable_admit": [True],  # 混合配置
    "enable_evict": [True],
}


@pytest.mark.parametrize("config", [
    ExecuteConfig(*v) for v in itertools.product(*params.values())
])
def test_table_to_filter_index_mapping_mixed_config(config: ExecuteConfig):
    """测试混合配置时的tableToFilterIndexMap映射关系"""
    mp.spawn(
        execute_with_custom_filter_config,
        args=(config, "mixed_config"),
        nprocs=WORLD_SIZE,
        join=True,
    )


params = {
    "world_size": [WORLD_SIZE],
    "table_num": [2],
    "embedding_dims": [[64, 64]],
    "num_embeddings": [[100, 200]],
    "sharding_type": ["row_wise"],
    "lookup_len": [16],
    "device": ["npu"],
    "enable_admit": [False],  # 所有表都不开启特征过滤
    "enable_evict": [False],
}


@pytest.mark.parametrize("config", [
    ExecuteConfig(*v) for v in itertools.product(*params.values())
])
def test_table_to_filter_index_mapping_no_filter(config: ExecuteConfig):
    """测试所有表都不开启特征过滤时的tableToFilterIndexMap映射关系"""
    mp.spawn(
        execute_with_custom_filter_config,
        args=(config, "no_filter"),
        nprocs=WORLD_SIZE,
        join=True,
    )


params = {
    "world_size": [WORLD_SIZE],
    "table_num": [1],  # 边界测试：单表
    "embedding_dims": [[128]],
    "num_embeddings": [[1000]],
    "sharding_type": ["row_wise"],
    "lookup_len": [64],
    "device": ["npu"],
    "enable_admit": [True],
    "enable_evict": [True],
}


@pytest.mark.parametrize("config", [
    ExecuteConfig(*v) for v in itertools.product(*params.values())
])
def test_table_to_filter_index_mapping_single_table(config: ExecuteConfig):
    """测试单表配置时的tableToFilterIndexMap映射关系"""
    mp.spawn(
        execute_with_custom_filter_config,
        args=(config, "single_table"),
        nprocs=WORLD_SIZE,
        join=True,
    )


def execute_with_custom_filter_config(rank: int, config: ExecuteConfig, filter_config_type: str):
    """执行带有自定义过滤器配置的测试"""
    world_size = config.world_size
    table_num = config.table_num
    embedding_dims = config.embedding_dims
    num_embeddings = config.num_embeddings
    sharding_type = config.sharding_type
    lookup_len = config.lookup_len
    device = config.device
    setup_logging(rank)
    logging.info("Testing tableToFilterIndexMap with filter_config_type: %s", filter_config_type)

    dataset = RandomRecDataset(BATCH_NUM, lookup_len, num_embeddings, table_num, is_evict_enabled=True)
    data_loader = DataLoader(
        dataset,
        batch_size=None,
        batch_sampler=None,
        pin_memory=True,
        pin_memory_device="npu",
        num_workers=1,
    )
    
    # 根据不同的过滤器配置类型设置表的准入淘汰配置
    embedding_configs = _create_embedding_configs_for_filter_test(
        table_num, embedding_dims, num_embeddings, filter_config_type
    )
    
    # 记录预期的tableToFilterIndexMap映射关系
    expected_filter_mapping = _calculate_expected_filter_mapping(embedding_configs)
    logging.info("Expected tableToFilterIndexMap for %s: %s", filter_config_type, expected_filter_mapping)
    
    test_model = TestModel(rank, world_size, device)
    test_results = test_model.test_loss(embedding_configs, data_loader, sharding_type, True, training=True)
    
    # 验证映射关系是否正确
    _verify_filter_mapping(embedding_configs, expected_filter_mapping, filter_config_type)
    
    dist.destroy_process_group()


def _create_embedding_configs_for_filter_test(table_num: int, embedding_dims: List[int], 
                                             num_embeddings: List[int], filter_config_type: str) -> List[EmbCacheEmbeddingConfig]:
    """根据测试类型创建不同配置的embedding configs"""
    embedding_configs = []
    default_config = AdmitAndEvictConfig()
    
    for i in range(table_num):
        if filter_config_type == "admit_only":
            # 只有第1和第3个表（索引0和2）开启准入
            admit_enabled = (i % 2 == 0)
            admit_threshold = 2 if admit_enabled else default_config.admit_threshold
            evict_threshold = default_config.evict_threshold
        elif filter_config_type == "evict_only":
            # 只有第2和第3个表（索引1和2）开启淘汰
            evict_enabled = (i >= 1)
            admit_threshold = default_config.admit_threshold
            evict_threshold = 2000_0000 if evict_enabled else default_config.evict_threshold
        elif filter_config_type == "mixed_config":
            # 混合配置：第1个表只开准入，第2个表只开淘汰，第3个表都开启，第4个表都不开启
            if i == 0:  # 第1个表：只开准入
                admit_threshold = 2
                evict_threshold = default_config.evict_threshold
            elif i == 1:  # 第2个表：只开淘汰
                admit_threshold = default_config.admit_threshold
                evict_threshold = 2000_0000
            elif i == 2:  # 第3个表：都开启
                admit_threshold = 2
                evict_threshold = 2000_0000
            else:  # 第4个表：都不开启
                admit_threshold = default_config.admit_threshold
                evict_threshold = default_config.evict_threshold
        elif filter_config_type == "no_filter":
            # 所有表都不开启特征过滤
            admit_threshold = default_config.admit_threshold
            evict_threshold = default_config.evict_threshold
        elif filter_config_type == "single_table":
            # 单表同时开启准入和淘汰
            admit_threshold = 2
            evict_threshold = 2000_0000
        else:
            raise ValueError(f"Unknown filter_config_type: {filter_config_type}")
            
        admit_and_evict_config = AdmitAndEvictConfig(
            admit_threshold=admit_threshold,
            not_admitted_default_value=0.999,
            evict_threshold=evict_threshold,
            evict_step_interval=EVICT_STEP_INTERVAL
        )
        
        ec_config = EmbCacheEmbeddingConfig(
            name=f"table{i}",
            embedding_dim=embedding_dims[i],
            num_embeddings=num_embeddings[i],
            feature_names=[f"feat{i}"],
            init_fn=weight_init,
            weight_init_min=0.0,
            weight_init_max=1.0,
            admit_and_evict_config=admit_and_evict_config
        )
        embedding_configs.append(ec_config)
    
    return embedding_configs


def _calculate_expected_filter_mapping(embedding_configs: List[EmbCacheEmbeddingConfig]) -> List[int]:
    """计算预期的tableToFilterIndexMap映射关系
    
    Returns:
        List[int]: 每个表索引对应的FeatureFilter索引，-1表示未开启特征过滤
    """
    expected_mapping = []
    filter_index = 0
    
    for i, config in enumerate(embedding_configs):
        # 根据代码逻辑：如果表开启了特征过滤（准入或淘汰），则分配一个FeatureFilter索引
        if config.admit_and_evict_config.IsFeatureFilterEnabled():
            expected_mapping.append(filter_index)
            filter_index += 1
        else:
            expected_mapping.append(-1)  # INVALID_KEY
    
    return expected_mapping


def _verify_filter_mapping(embedding_configs: List[EmbCacheEmbeddingConfig], 
                          expected_mapping: List[int], filter_config_type: str):
    """验证tableToFilterIndexMap映射关系是否符合预期"""
    logging.info("Verifying filter mapping for %s", filter_config_type)
    
    # 打印详细的表配置信息
    _print_table_configuration_details(embedding_configs, expected_mapping, filter_config_type)
    
    # 验证映射长度
    assert len(expected_mapping) == len(embedding_configs), \
        f"Mapping length mismatch: expected {len(embedding_configs)}, got {len(expected_mapping)}"
    
    # 验证映射关系的逻辑正确性
    filter_enabled_count = 0
    for i, config in enumerate(embedding_configs):
        is_filter_enabled = config.admit_and_evict_config.IsFeatureFilterEnabled()
        expected_filter_index = expected_mapping[i]
        
        if is_filter_enabled:
            # 开启特征过滤的表应该有有效的FeatureFilter索引
            assert expected_filter_index >= 0, \
                f"Table {i} has feature filter enabled but got invalid filter index {expected_filter_index}"
            filter_enabled_count += 1
        else:
            # 未开启特征过滤的表应该映射到-1（INVALID_KEY）
            assert expected_filter_index == -1, \
                f"Table {i} has feature filter disabled but got valid filter index {expected_filter_index}"
    
    # 验证FeatureFilter索引的连续性
    if filter_enabled_count > 0:
        filter_indices = [idx for idx in expected_mapping if idx >= 0]
        filter_indices.sort()
        expected_indices = list(range(filter_enabled_count))
        assert filter_indices == expected_indices, \
            f"Filter indices should be consecutive starting from 0: expected {expected_indices}, got {filter_indices}"
    
    # 验证特定配置类型的预期结果
    if filter_config_type == "admit_only":
        # admit_only配置：只有第1和第3个表（索引0和2）应该开启
        for i, expected_idx in enumerate(expected_mapping):
            if i % 2 == 0:  # 第1和第3个表
                assert expected_idx >= 0, f"Table {i} should have filter enabled in admit_only config"
            else:  # 第2个表
                assert expected_idx == -1, f"Table {i} should have filter disabled in admit_only config"
    
    elif filter_config_type == "evict_only":
        # evict_only配置：只有第2和第3个表（索引1和2）应该开启
        for i, expected_idx in enumerate(expected_mapping):
            if i >= 1:  # 第2和第3个表
                assert expected_idx >= 0, f"Table {i} should have filter enabled in evict_only config"
            else:  # 第1个表
                assert expected_idx == -1, f"Table {i} should have filter disabled in evict_only config"
    
    elif filter_config_type == "mixed_config":
        # mixed_config配置：第1、2、3个表开启，第4个表不开启
        for i, expected_idx in enumerate(expected_mapping):
            if i <= 2:  # 前3个表
                assert expected_idx >= 0, f"Table {i} should have filter enabled in mixed_config"
            else:  # 第4个表
                assert expected_idx == -1, f"Table {i} should have filter disabled in mixed_config"
    
    elif filter_config_type == "no_filter":
        # no_filter配置：所有表都不开启
        for i, expected_idx in enumerate(expected_mapping):
            assert expected_idx == -1, f"Table {i} should have filter disabled in no_filter config"
    
    elif filter_config_type == "single_table":
        # single_table配置：唯一的表应该开启
        assert len(expected_mapping) == 1, f"Single table config should have exactly 1 table"
        assert expected_mapping[0] == 0, f"Single table should have filter index 0, got {expected_mapping[0]}"
    
    logging.info("Filter mapping verification passed for %s", filter_config_type)


def _print_table_configuration_details(embedding_configs: List[EmbCacheEmbeddingConfig],
                                      expected_mapping: List[int], filter_config_type: str):
    """打印表配置的详细信息，帮助调试和验证"""
    logging.info("=== Table Configuration Details for %s ===", filter_config_type)
    logging.info("Total tables: %d", len(embedding_configs))
    logging.info("Expected tableToFilterIndexMap: %s", expected_mapping)
    
    filter_enabled_tables = []
    for i, config in enumerate(embedding_configs):
        admit_config = config.admit_and_evict_config
        is_admit_enabled = admit_config.IsAdmitEnabled()
        is_evict_enabled = admit_config.IsEvictEnabled()
        is_filter_enabled = admit_config.IsFeatureFilterEnabled()
        
        table_info = {
            "table_index": i,
            "table_name": config.name,
            "admit_enabled": is_admit_enabled,
            "evict_enabled": is_evict_enabled,
            "filter_enabled": is_filter_enabled,
            "expected_filter_index": expected_mapping[i],
            "admit_threshold": admit_config.admitThreshold if is_admit_enabled else "N/A",
            "evict_threshold": admit_config.evictThreshold if is_evict_enabled else "N/A"
        }
        
        logging.info("Table %d (%s): admit=%s, evict=%s, filter=%s, filter_index=%d, "
                    "admit_threshold=%s, evict_threshold=%s",
                    i, config.name, is_admit_enabled, is_evict_enabled, is_filter_enabled,
                    expected_mapping[i], table_info["admit_threshold"], table_info["evict_threshold"])
        
        if is_filter_enabled:
            filter_enabled_tables.append(i)
    
    logging.info("Filter enabled table indices: %s", filter_enabled_tables)
    logging.info("Expected FeatureFilter count: %d", len(filter_enabled_tables))
    
    # 验证INVALID_KEY的使用是否符合预期
    invalid_key_count = sum(1 for idx in expected_mapping if idx == -1)
    valid_filter_count = sum(1 for idx in expected_mapping if idx >= 0)
    logging.info("Tables with INVALID_KEY (-1) mapping: %d", invalid_key_count)
    logging.info("Tables with valid filter mapping: %d", valid_filter_count)
    
    # 验证映射的一致性
    expected_filter_count = len(filter_enabled_tables)
    assert valid_filter_count == expected_filter_count, \
        f"Mismatch in filter count: expected {expected_filter_count}, got {valid_filter_count}"
    
    logging.info("=== End of Table Configuration Details ===")


if __name__ == '__main__':
    # 运行tableToFilterIndexMap相关的测试
    logging.basicConfig(level=logging.INFO)
    
    # 测试混合配置
    test_table_to_filter_index_mapping_mixed_config(ExecuteConfig(
        world_size=WORLD_SIZE,
        table_num=4,
        embedding_dims=[32, 64, 128, 256],
        num_embeddings=[800, 1600, 3200, 6400],
        sharding_type="row_wise",
        lookup_len=32,
        device="npu",
        enable_admit=True,
        enable_evict=True
    ))
    
    # 测试单表配置
    test_table_to_filter_index_mapping_single_table(ExecuteConfig(
        world_size=WORLD_SIZE,
        table_num=1,
        embedding_dims=[128],
        num_embeddings=[1000],
        sharding_type="row_wise",
        lookup_len=64,
        device="npu",
        enable_admit=True,
        enable_evict=True
    ))
    
    # 测试无过滤器配置
    test_table_to_filter_index_mapping_no_filter(ExecuteConfig(
        world_size=WORLD_SIZE,
        table_num=2,
        embedding_dims=[64, 64],
        num_embeddings=[100, 200],
        sharding_type="row_wise",
        lookup_len=16,
        device="npu",
        enable_admit=False,
        enable_evict=False
    ))
