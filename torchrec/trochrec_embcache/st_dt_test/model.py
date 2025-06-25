#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import logging
import os
import torch
import torchrec
import torch_npu

from typing import Dict, List
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import ModuleList

from torchrec import (
    EmbeddingConfig,
    EmbeddingBagConfig,
    EmbeddingBagCollection,
    EmbeddingCollection,
    KeyedJaggedTensor,
)
import torchrec.distributed
from torch.optim import Adam, Adagrad
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    Topology,
    ParameterConstraints,
)
from torchrec.distributed.types import ShardingEnv
from torchrec.optim.keyed import CombinedOptimizer
from torchrec_embcache.distributed.embedding import EmbCacheEmbeddingCollection
from torchrec_embcache.distributed.embedding_bag import EmbCacheEmbeddingBagCollection
from torchrec_embcache.distributed.train_pipeline import EmbCacheTrainPipelineSparseDist
from torchrec_embcache.distributed.sharding.embedding_sharder import (
EmbCacheEmbeddingCollectionSharder, 
EmbCacheEmbeddingBagCollectionSharder
)
from torchrec_embcache.distributed.modules.cache_embedding_configs import (
AdmitAndEvictConfig, 
EmbCacheEmbeddingConfig,
EmbCacheEmbeddingBagConfig,
InitializerType
)

from dataset import Batch
from util import logging


lib_fbgemm_npu_api_so_path = os.getenv('LIB_FBGEMM_NPU_API_SO_PATH')
torch.ops.load_library(lib_fbgemm_npu_api_so_path)


OPTIMIZER_PARAM = {
    Adam: dict(lr=0.02),
    Adagrad: dict(lr=0.02, eps=1.0e-8),
}


def permute_values(kjt: KeyedJaggedTensor, feature_names_lst) -> torch.Tensor:
    values = []
    jt_dict = kjt.to_dict()
    for feature_name in itertools.chain(*feature_names_lst):
        jt = jt_dict[feature_name]
        values.append(jt)
    values = torch.concat(values, dim=1)
    return values

# ec和ebc查询结果返回数据类型不一样
def permute_values_ec(result: Dict, feature_num_lst) -> torch.Tensor:
    values = []
    for feature_name in itertools.chain(*feature_num_lst):
        jt = result[feature_name].values()
        values.append(jt)
    values = torch.concat(values, dim=1)
    return values

class Model(torch.nn.Module):
    def __init__(self, module, feature_names_lst):
        super().__init__()
        self._module = module
        self.feature_names_lst = feature_names_lst

    @property
    def ebc(self):
        return self._module

    def forward(self, batch: Batch):
        results = []
        for i, module in enumerate(self._module):
            result = module(getattr(batch, f"instance{i}_sparse_features"))
            result = permute_values(result, self.feature_names_lst)
            results.append(result)

        result = torch.concat(results, dim=1)
        loss = result.sum()
        return loss, result


class ModelEc(torch.nn.Module):
    def __init__(self, module, feature_names_lst):
        super().__init__()
        self._module = module
        self.feature_names_lst = feature_names_lst

    @property
    def ec(self):
        return self._module

    def forward(self, batch: Batch):
        results = []
        for i, module in enumerate(self._module):
            result = module(getattr(batch, f"instance{i}_sparse_features"))
            result = permute_values_ec(result, self.feature_names_lst)
            results.append(result)

        result = torch.concat(results, dim=1)
        loss = result.sum()
        return loss, result


COLLECTION_DICT = {
    "ec": {
        "collection": EmbCacheEmbeddingCollection,
        "collection_cpu": EmbeddingCollection,
        "sharder": EmbCacheEmbeddingCollectionSharder,
        "model": ModelEc,
        "config": EmbCacheEmbeddingConfig
    },
    "ebc": {
        "collection": EmbCacheEmbeddingBagCollection,
        "collection_cpu": EmbeddingBagCollection,
        "sharder": EmbCacheEmbeddingBagCollectionSharder,
        "model": Model,
        "config": EmbCacheEmbeddingBagConfig
    }
}


class TestModel:
    def __init__(self, rank, world_size, device, instances, feature_names_lst, BATCH_NUM, collection_type="ec"):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.instances = instances
        self.feature_names_lst = feature_names_lst
        self.pg_method = "hccl" if device == "npu" else "gloo"
        if device == "npu":
            torch_npu.npu.set_device(rank)
        self.BATCH_NUM = BATCH_NUM
        self.setup(rank=rank, world_size=world_size)
        if collection_type not in ["ec", "ebc"]:
            raise ValueError(f"collection type must be one of ec or ebc, find {collection_type} instead")
        self.collection_type = collection_type
        self.module = None
        self.ddpModel = None
        self.npu_device: torch.device = torch.device(f"npu:{rank}")
        self.cpu_device = torch.device("cpu")
        self.table_num = 0

    def cpu_golden_loss(
        self, 
        embedding_config,
        dataloader: DataLoader[Batch], 
        optim
    ):
        pg = dist.new_group(backend="gloo")
        table_num = len(embedding_config)
        module = ModuleList()
        for _ in range(self.instances):
            module.append(COLLECTION_DICT[self.collection_type]["collection_cpu"](device="cpu", tables=embedding_config))
        module = COLLECTION_DICT[self.collection_type]["model"](module, self.feature_names_lst)
        model = DDP(module, device_ids=None, process_group=pg)
        opt = optim(module.parameters(), **OPTIMIZER_PARAM[optim])

        results = []
        batch: Batch
        iter_ = iter(dataloader)
        for _ in range(self.BATCH_NUM):
            batch = next(iter_)
            opt.zero_grad()
            loss, out = model(batch)
            results.append(loss.detach().cpu())
            results.append(out.detach().cpu())
            loss.backward()
            opt.step()
        for i, instance in enumerate(getattr(module, self.collection_type)):
            for j in range(table_num):
                logging.debug(
                    "cpu after train instance%d single table%d weight %s",
                    i,
                    j,
                    instance.embeddings[f"table{i}"].weight if self.collection_type == "ec"
                    else instance.embedding_bags[f"table{i}"].weight,
                )
        logging.debug(f"cpu result: {results}")
        return results

    def setup(self, rank: int, world_size: int):
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "6000"
        dist.init_process_group(self.pg_method, rank=rank, world_size=world_size)
        os.environ["LOCAL_RANK"] = f"{rank}"

    def init_ddp_model(
        self,
        embedding_config,
        sharding_type: str,
        optim,
        lookup_lens,
    ):
        rank, world_size = self.rank, self.world_size
        host_gp = dist.new_group(backend="gloo")
        host_env = ShardingEnv(world_size=world_size, rank=rank, pg=host_gp)
        # Shard
        self.table_num = len(embedding_config)
        module = ModuleList()
        for _ in range(self.instances):
            module.append(COLLECTION_DICT[self.collection_type]["collection"](device=torch.device("meta"), tables=embedding_config,
                                                      batch_size=lookup_lens, multi_hot_sizes=[1]*self.table_num,
                                                      world_size=dist.get_world_size()))
        module = COLLECTION_DICT[self.collection_type]["model"](module, self.feature_names_lst)
        apply_optimizer_in_backward(
            optimizer_class=optim,
            params=module.parameters(),
            optimizer_kwargs=OPTIMIZER_PARAM[optim],
        )
        # Shard
        constrans = {
            f"table{i}": ParameterConstraints(sharding_types=[sharding_type])
            for i in range(self.table_num)
        }
        rank = int(os.environ["LOCAL_RANK"])
        
        cpu_pg = dist.new_group(backend="gloo")
        cpu_env = ShardingEnv.from_process_group(cpu_pg)
        hash_shader = COLLECTION_DICT[self.collection_type]["sharder"](
            cpu_device=self.cpu_device,
            cpu_env=cpu_env,
            npu_device=self.npu_device,
            npu_env=ShardingEnv.from_process_group(dist.GroupMember.WORLD),
        )
        shaders = [hash_shader]
        planner = EmbeddingShardingPlanner(
            topology=Topology(world_size=self.world_size, compute_device=self.device),
            constraints=constrans,
        )
        plan = planner.collective_plan(
            module, shaders, dist.GroupMember.WORLD
        )
        if self.rank == 0:
            logging.debug(plan)

        ddpModel = torchrec.distributed.DistributedModelParallel(
            module,
            sharders=shaders,
            device=self.npu_device,
            plan=plan,
        )
        self.ddpModel = ddpModel
        self.module = module
        logging.debug(ddpModel)

    def test_pipe_loss(self,
        dataloader: DataLoader[Batch],):
        # Optimizer
        optimizer = CombinedOptimizer([self.ddpModel.fused_optimizer])
        results = []
        iter_ = iter(dataloader)
        self.ddpModel.train()
        pipe = EmbCacheTrainPipelineSparseDist(
            self.ddpModel,
            optimizer=optimizer,
            cpu_device=self.cpu_device,
            npu_device=self.npu_device,
            return_loss=True,
        )
        for _ in range(self.BATCH_NUM):
            out, loss = pipe.progress(iter_)
            results.append(loss.detach().cpu())
            results.append(out.detach().cpu())
        for i, instance in enumerate(getattr(self.module, self.collection_type)):
            for j in range(self.table_num):
                logging.debug(
                    "npu after train instance%d single table%d weight %s",
                    i,
                    j,
                    instance.embeddings[f"table{i}"].weight if self.collection_type == "ec"
                    else instance.embedding_bags[f"table{i}"].weight,
                )
        logging.debug(f"npu results: {results}")
        return results


def generate_hash_config(
    embedding_dims, num_embeddings, pool_type, feature_names, init_fn, collection_type
):
    test_table_configs: List[COLLECTION_DICT[collection_type]["collection_cpu"]] = []
    for i, (table_dim, num_embedding, feature_name) in enumerate(zip(embedding_dims, num_embeddings, feature_names)):
        if collection_type == "ebc":    
            config = COLLECTION_DICT[collection_type]["config"](
                name=f"table{i}",
                embedding_dim=table_dim,
                num_embeddings=num_embedding,
                feature_names=feature_name,
                pooling=pool_type,
                init_fn=init_fn,
                weight_init_max=1.0,
                weight_init_min=0.0,
                initializer_type=InitializerType.LINEAR
            )
        else:
            admit_and_evict_config = AdmitAndEvictConfig(admit_threshold=-1, 
                                                         not_admitted_default_value=0.99)
            config = COLLECTION_DICT[collection_type]["config"](
                name=f"table{i}",
                embedding_dim=table_dim,
                num_embeddings=num_embedding,
                feature_names=feature_name,
                init_fn=init_fn,
                weight_init_max=1.0,
                weight_init_min=0.0,
                admit_and_evict_config=admit_and_evict_config,
                initializer_type=InitializerType.LINEAR
            )
        test_table_configs.append(config)
    return test_table_configs