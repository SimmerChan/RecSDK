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
from torchrec.distributed.types import EmbeddingBagCollectionContext
from torchrec.sparse.jagged_tensor import KeyedTensor


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

def test_output_dist(hybrid_sharded_embedding_bag_collection):
    # Mock context
    ctx = EmbeddingBagCollectionContext()
    ctx.sharding_contexts = [None, None]

    # Mock output tensors
    output = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
    ]

    # Call the output_dist method
    awaitable = hybrid_sharded_embedding_bag_collection.output_dist(ctx, output)

    # Wait for the result
    result = awaitable.wait()

    # Assertions
    assert isinstance(result, KeyedTensor)
    assert result.values().shape == (2, 4)  # Verify the shape of the output tensor
    assert result.keys() == hybrid_sharded_embedding_bag_collection._embedding_names
