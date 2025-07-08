#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch
from hybrid_torchrec.distributed.embeddingbag import HybridShardedEmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KJTList
from torchrec.distributed.types import EmbeddingBagCollectionAwaitable


@pytest.fixture
def hybrid_sharded_embedding_bag_collection():
    # Mock setup for HybridShardedEmbeddingBagCollection
    table_name_to_parameter_sharding = {
        "table1": {"sharding_type": "row_wise"},
        "table2": {"sharding_type": "table_wise"},
    }
    env = None  # Mock environment
    host_env = None  # Mock host environment
    embedding_bag_configs = [
        {"name": "table1", "embedding_dim": 128, "num_embeddings": 1000, "feature_names": ["feature1"]},
        {"name": "table2", "embedding_dim": 64, "num_embeddings": 500, "feature_names": ["feature2"]},
    ]
    module = None  # Mock module

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


def test_post_input_dist(hybrid_sharded_embedding_bag_collection):
    # Mock input features
    features = KJTList([
        KeyedJaggedTensor(
            keys=["feature1"],
            values=torch.tensor([1, 2, 3], dtype=torch.float32),
            lengths=torch.tensor([3], dtype=torch.int32),
        ),
        KeyedJaggedTensor(
            keys=["feature2"],
            values=torch.tensor([4, 5, 6], dtype=torch.float32),
            lengths=torch.tensor([3], dtype=torch.int32),
        ),
    ])

    # Mock context
    ctx = EmbeddingBagCollectionAwaitable()

    # Call the post_input_dist method
    awaitable = hybrid_sharded_embedding_bag_collection.post_input_dist(ctx, features)

    # Wait for the result
    result = awaitable.wait()

    # Assertions
    assert len(result) == len(hybrid_sharded_embedding_bag_collection._post_input_dists)
    for shard_result in result:
        assert isinstance(shard_result, KeyedJaggedTensor)
