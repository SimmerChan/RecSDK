#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import unittest
import torch
import torch.distributed as dist
from torchrec.distributed.embedding_types import (
    BaseGroupedFeatureProcessor,
    EmbeddingComputeKernel,
    GroupedEmbeddingConfig,
    BaseEmbeddingLookup,
)
from torchrec.distributed.types import ShardingType
torch_npu = pytest.importorskip("torch_npu", reason="NPU设备支持未安装")

from hybrid_torchrec.distributed import HybridGroupedPooledEmbeddingsLookup


class TestHybridGroupedPooledEmbeddingsLookup:
    # def __init__(self):
    #     self.device = "cpu"
    #     self.pg = dist.new_group([0]) if dist.is_initialized() else None

    def test_fused_kernel_initialization(self):
        config = GroupedEmbeddingConfig(
            num_features=4,
            compute_kernel=EmbeddingComputeKernel.FUSED,
            embedding_dim=64
        )
        lookup = HybridGroupedPooledEmbeddingsLookup(
            grouped_configs=[config],
            device="cpu",
            pg=None,
            sharding_type=ShardingType.TABLE_WISE
        )
        assert len(lookup._emb_modules) == 1
        # self.assertEqual(len(lookup._emb_modules), 1)
        # self.assertTrue(hasattr(lookup, "_dummy_embs_tensor"))

    # def test_keyvalue_kernel_initialization(self):
    #     config = GroupedEmbeddingConfig(
    #         num_features=2,
    #         compute_kernel=EmbeddingComputeKernel.KEY_VALUE,
    #         embedding_dim=32
    #     )
    #     lookup = HybridGroupedPooledEmbeddingsLookup(
    #         grouped_configs=[config],
    #         device="cpu",
    #         pg=None
    #     )
    #     # self.assertEqual(lookup._feature_splits, [2])
    #
    # def test_unsupported_kernel_raises(self):
    #     config = GroupedEmbeddingConfig(
    #         num_features=3,
    #         compute_kernel="UNSUPPORTED",
    #         embedding_dim=16
    #     )
    #     with pytest.raises(ValueError):
    #         HybridGroupedPooledEmbeddingsLookup(
    #             grouped_configs=[config],
    #             device="cpu"
    #         )
    #
    # def test_empty_configs_handling(self):
    #     lookup = HybridGroupedPooledEmbeddingsLookup(
    #         grouped_configs=[],
    #         device="cpu"
    #     )
    #     # self.assertEqual(lookup._dummy_embs_tensor.shape, torch.Size([0]))


# if __name__ == "__main__":
#     unittest.main()
