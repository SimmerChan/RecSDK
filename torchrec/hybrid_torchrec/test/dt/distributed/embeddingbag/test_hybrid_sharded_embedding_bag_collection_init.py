#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from hybrid_torchrec.distributed.embeddingbag import HybridShardedEmbeddingBagCollection
from torchrec.distributed.types import ParameterSharding, ShardingEnv
from torchrec.modules.embedding_configs import EmbeddingBagConfig, PoolingType
from torchrec.modules.embedding_modules import EmbeddingBagCollection


def test_hybrid_sharded_embedding_bag_collection_init():
    table_name_to_parameter_sharding = {
        "table1": ParameterSharding(
            sharding_types=["row_wise"], compute_kernels=["fused"]
        ),
        "table2": ParameterSharding(
            sharding_types=["table_wise"], compute_kernels=["dense"]
        ),
    }
    env = ShardingEnv(world_size=1, rank=0)
    host_env = ShardingEnv(world_size=1, rank=0)
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
    module = EmbeddingBagCollection(tables=embedding_bag_configs, device="cpu")

    # Initialize the HybridShardedEmbeddingBagCollection
    hybrid_sharded_ebc = HybridShardedEmbeddingBagCollection(
        module=module,
        table_name_to_parameter_sharding=table_name_to_parameter_sharding,
        env=env,
        host_env=host_env,
        fused_params=None,
        device=torch.device("cpu"),
        qcomm_codecs_registry=None,
        module_fqn=None,
    )

    # Assertions to verify initialization
    assert hybrid_sharded_ebc._env == env
    assert hybrid_sharded_ebc._host_env == host_env
    assert hybrid_sharded_ebc._device == torch.device("cpu")
    assert hybrid_sharded_ebc._embedding_bag_configs == embedding_bag_configs
    assert hybrid_sharded_ebc._table_names == ["table1", "table2"]
    assert hybrid_sharded_ebc._table_name_to_config["table1"].name == "table1"
    assert hybrid_sharded_ebc._table_name_to_config["table2"].name == "table2"
