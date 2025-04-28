#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import torch
from typing import List
import torch_npu
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torchrec
import pytest
import logging
from torchrec import (
    EmbeddingConfig,
)
import torchrec.distributed
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    Topology,
    ParameterConstraints,
)
from torchrec.distributed.types import ShardingEnv
from torchrec.optim.keyed import CombinedOptimizer
from hybrid_torchrec import HashEmbeddingCollection, HashEmbeddingConfig
from hybrid_torchrec.distributed.sharding_plan import get_default_hybrid_sharders
from model import Model
from dataset import RandomRecDataset, Batch
from util import setup_logging

LOOP_TIMES = 8
BATCH_NUM = 32
WORLD_SIZE = 2
torch.ops.load_library("/home/t30062856/code/libfbgemm_npu_api.so")

def generate_hash_config(embedding_dims: List[int],
                         num_embeddings: List[int]) -> List[HashEmbeddingConfig]:
    test_table_configs: List[HashEmbeddingCollection] = []
    for i, (table_dim, num_embedding) in enumerate(zip(embedding_dims, num_embeddings)):
        config = HashEmbeddingConfig(
            name=f"table{i}",
            embedding_dim=table_dim,
            num_embeddings=num_embedding,
            feature_names=[f"feat{i}"],
            init_fn=weight_init,
        )
        test_table_configs.append(config)
    return test_table_configs


def execute(
    rank,
    world_size,
    table_num,
    embedding_dims,
    num_embeddings,
    sharding_type,
    lockup_len,
    device,
):
    setup_logging(rank)
    logging.info("this test %s", os.path.basename(__file__))
    embeding_config = generate_hash_config(embedding_dims, num_embeddings)

    dataset = RandomRecDataset(BATCH_NUM, lockup_len, num_embeddings, table_num)
    gloden_dataset_loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=None,
        pin_memory=True,
        pin_memory_device="npu",
        num_workers=1,
    )

    test_model = TestModel(rank, world_size, device)

    gloden_results = test_model.cpu_gloden_loss(embeding_config, gloden_dataset_loader)
    test_results = test_model.test_loss(embeding_config, data_loader, sharding_type)
    for gloden, result in zip(gloden_results, test_results):
        logging.debug("")
        logging.debug("===========================")
        logging.debug("result test %s", gloden)
        logging.debug("gloden test %s", result)
        assert torch.allclose(
            gloden, result, rtol=1e-04, atol=1e-04
        ), "gloden and result is not closed"


def weight_init(param: torch.nn.Parameter):
    if len(param.shape) != 2:
        return
    torch.manual_seed(param.shape[1])
    result = torch.randn((1, param.shape[1])).repeat(param.shape[0], 1)
    param.data.copy_(result)


class TestModel:
    def __init__(self, rank, world_size, device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.pg_method = "hccl" if device == "npu" else "gloo"
        if device == "npu":
            torch_npu.npu.set_device(rank)
        self.setup(rank=rank, world_size=world_size)

    @staticmethod
    def cpu_gloden_loss(
        embeding_config: List[EmbeddingConfig], dataloader: DataLoader[Batch]
    ):
        pg = dist.new_group(backend="gloo")
        table_num = len(embeding_config)
        ec = HashEmbeddingCollection(device="cpu", tables=embeding_config)

        num_features = sum([c.num_features() for c in embeding_config])
        ec = Model(ec, num_features)
        model = DDP(ec, device_ids=None, process_group=pg)

        opt = torch.optim.Adagrad(ec.parameters(), lr=0.02, eps=1e-8)
        results = []
        batch: Batch
        iter_ = iter(dataloader)
        for _ in range(LOOP_TIMES):
            batch = next(iter_)
            opt.zero_grad()
            loss, out = model(batch)
            results.append(loss.detach().cpu())
            results.append(out.detach().cpu())
            loss.backward()
            opt.step()

        for i in range(table_num):
            logging.debug(
                "single table%d weight %s",
                i,
                ec.ec.embeddings[f"table{i}"].weight,
            )
        return results

    def setup(self, rank: int, world_size: int):
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "6000"
        dist.init_process_group(self.pg_method, rank=rank, world_size=world_size)

    def test_loss(
        self,
        embeding_config: List[EmbeddingConfig],
        dataloader: DataLoader[Batch],
        sharding_type: str,
    ):
        num_features = sum([c.num_features() for c in embeding_config])
        rank, world_size = self.rank, self.world_size
        host_gp = dist.new_group(backend="gloo")
        host_env = ShardingEnv(world_size=world_size, rank=rank, pg=host_gp)
        # Shard
        table_num = len(embeding_config)
        ec = HashEmbeddingCollection(device=torch.device("meta"), tables=embeding_config)
        ec = Model(ec, num_features)
        apply_optimizer_in_backward(
            optimizer_class=torch.optim.Adagrad,
            params=ec.parameters(),
            optimizer_kwargs={"lr": 0.02},
        )
        # Shard
        constrans = {
            f"table{i}": ParameterConstraints(sharding_types=[sharding_type])
            for i in range(table_num)
        }
        planner = EmbeddingShardingPlanner(
            topology=Topology(world_size=self.world_size, compute_device=self.device),
            constraints=constrans,
        )
        plan = planner.collective_plan(
            ec, get_default_hybrid_sharders(host_env), dist.GroupMember.WORLD
        )
        if self.rank == 0:
            logging.debug(plan)

        ddpModel = torchrec.distributed.DistributedModelParallel(
            ec,
            sharders=get_default_hybrid_sharders(host_env),
            device=torch.device(self.device),
            plan=plan,
        )
        logging.debug(ddpModel)
        # Optimizer
        optimizer = CombinedOptimizer([ddpModel.fused_optimizer])
        results = []
        batch: Batch
        iter_ = iter(dataloader)
        for _ in range(LOOP_TIMES):
            batch = next(iter_).to(self.device)
            optimizer.zero_grad()
            loss, out = ec(batch)
            results.append(loss.detach().cpu())
            results.append(out.detach().cpu())
            loss.backward()
            optimizer.step()

        for i in range(table_num):
            logging.debug(
                "shard table%d weight %s",
                i,
                ddpModel.module.ec.embeddings[f"table{i}"].weight,
            )
        return results


@pytest.mark.parametrize("table_num", [2])
@pytest.mark.parametrize("embedding_dims", [[32, 64, 128]])
@pytest.mark.parametrize("num_embeddings", [[400, 4000, 400]])
@pytest.mark.parametrize("sharding_type", ["table_wise", "row_wise"])
@pytest.mark.parametrize("lockup_len", [1024])
@pytest.mark.parametrize("device", ["cpu", "npu"])
def test_hybrid_hash_embedding(
    table_num,
    embedding_dims,
    num_embeddings,
    sharding_type,
    lockup_len,
    device,
):
    if device == "cpu" and sharding_type == "row_wise":
        return
    mp.spawn(
        execute,
        args=(
            WORLD_SIZE,
            table_num,
            embedding_dims,
            num_embeddings,
            sharding_type,
            lockup_len,
            device,
        ),
        nprocs=WORLD_SIZE,
        join=True,
    )
