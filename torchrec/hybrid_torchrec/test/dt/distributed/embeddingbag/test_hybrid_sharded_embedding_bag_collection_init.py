#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from unittest.mock import MagicMock

import torch
import torchrec
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology, ParameterConstraints
from torchrec.distributed.types import ShardingEnv
from torchrec.modules.embedding_configs import EmbeddingBagConfig, PoolingType
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from hybrid_torchrec.distributed import get_default_hybrid_sharders
from hybrid_torchrec.distributed.embeddingbag import HybridShardedEmbeddingBagCollection

DEVICE = torch.device("cpu")


def set_env(monkeypatch):
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "5678")

    torchrec.distributed.planner.ParameterConstraints.__post_init__ = MagicMock(return_value=None)
    torchrec.tensor_types.check = MagicMock(return_value=None)
    torchrec.distributed.model_parallel.check = MagicMock(return_value=None)
    torchrec.distributed.planner.types.check = MagicMock(return_value=None)

    torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
    pg = torch.distributed.group.WORLD
    env = ShardingEnv(world_size=1, rank=0, pg=pg)
    host_env = ShardingEnv(world_size=1, rank=0, pg=pg)
    return env, host_env


def create_ebc():
    embedding_bag_configs = [
        EmbeddingBagConfig(
            name="table1",
            embedding_dim=128,
            num_embeddings=1000,
            feature_names=["feature1"],
            pooling=PoolingType.SUM,
        ),
        EmbeddingBagConfig(
            name="table2",
            embedding_dim=64,
            num_embeddings=500,
            feature_names=["feature2"],
            pooling=PoolingType.MEAN,
        ),
    ]
    return EmbeddingBagCollection(tables=embedding_bag_configs, device=DEVICE), embedding_bag_configs


def create_planner():
    constraints = {
        "table1": ParameterConstraints(
            sharding_types=["row_wise"], compute_kernels=["fused"],
        ),
        "table2": ParameterConstraints(
            sharding_types=["row_wise"], compute_kernels=["fused"],
        ),
    }

    planner = EmbeddingShardingPlanner(
        topology=Topology(world_size=1, compute_device=DEVICE.type),
        constraints=constraints,
    )
    return planner


def test_hybrid_sharded_embedding_bag_collection_init(monkeypatch):
    env, host_env = set_env(monkeypatch)
    module, embedding_bag_configs = create_ebc()
    planner = create_planner()
    hybrid_sharder = get_default_hybrid_sharders(host_env=host_env)

    plan = planner.collective_plan(module, hybrid_sharder, torch.distributed.GroupMember.WORLD)
    sharded_params = plan.get_plan_for_module(list(plan.plan.keys())[0])

    # Initialize the HybridShardedEmbeddingBagCollection
    hybrid_sharded_ebc = HybridShardedEmbeddingBagCollection(
        module=module,
        table_name_to_parameter_sharding=sharded_params,
        env=env,
        host_env=host_env,
        fused_params=None,
        device=DEVICE,
        qcomm_codecs_registry=None,
        module_fqn=None,
    )

    # Assertions to verify initialization
    assert hybrid_sharded_ebc._env == env
    assert hybrid_sharded_ebc._host_env == host_env
    assert hybrid_sharded_ebc._device == DEVICE
    assert hybrid_sharded_ebc._embedding_bag_configs == embedding_bag_configs
    assert hybrid_sharded_ebc._table_names == ["table1", "table2"]
    assert hybrid_sharded_ebc._table_name_to_config["table1"].name == "table1"
    assert hybrid_sharded_ebc._table_name_to_config["table2"].name == "table2"
