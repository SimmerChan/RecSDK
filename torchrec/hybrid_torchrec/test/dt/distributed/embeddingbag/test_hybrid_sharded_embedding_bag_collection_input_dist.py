#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch
from torchrec.distributed.types import ParameterSharding, ShardingEnv
from torchrec.modules.embedding_configs import EmbeddingBagConfig, PoolingType
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from hybrid_torchrec.distributed.embeddingbag import HybridShardedEmbeddingBagCollection


@pytest.fixture
def hybrid_sharded_embedding_bag_collection():
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

    return HybridShardedEmbeddingBagCollection(
        module=module,
        table_name_to_parameter_sharding=table_name_to_parameter_sharding,
        env=env,
        host_env=host_env,
        fused_params=None,
        device=torch.device("cpu"),
        qcomm_codecs_registry=None,
        module_fqn=None,
    )


def test_input_dist(hybrid_sharded_embedding_bag_collection):
    # Create a mock KeyedJaggedTensor input
    features = KeyedJaggedTensor(
        keys=["feature1", "feature2"],
        values=torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32),
        lengths=torch.tensor([3, 3], dtype=torch.int32),
    )

    # Call the input_dist method
    ctx = hybrid_sharded_embedding_bag_collection.create_context()
    awaitable = hybrid_sharded_embedding_bag_collection.input_dist(ctx, features)

    # Wait for the result
    result = awaitable.wait()

    # Assertions
    assert len(result) == len(hybrid_sharded_embedding_bag_collection._input_dists)
    for shard_result in result:
        assert isinstance(shard_result, KeyedJaggedTensor)
