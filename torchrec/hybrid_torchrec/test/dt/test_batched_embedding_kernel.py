import pytest
import torch
import unittest
from typing import List, Optional

from fbgemm_gpu.split_embedding_configs import EmbOptimType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    EmbeddingLocation,
    PoolingMode,
)

from hybrid_torchrec.distributed.batched_embedding_kernel import ( 
    HybridSplitTableBatchedEmbeddingBagsCodegen,
    HybridBatchedFusedEmbeddingBag,
)
import torch.distributed as dist
from torchrec import ComputeDevice, PoolingType, DataType, ShardedEmbeddingTable, ShardingType
from torchrec.distributed import GroupedEmbeddingConfig
from torch.optim import Adam, Adagrad, SGD
from parameterized import parameterized
DEVICEID = "npu:0"

TORCH_OPTIMIZER_TO_FBGEMM = {
    Adam: EmbOptimType.ADAM,
    Adagrad: EmbOptimType.EXACT_ADAGRAD,
    SGD: EmbOptimType.EXACT_SGD
}
OPTIMIZER_PARAM = {
    Adam: dict(lr=0.01),
    Adagrad: dict(lr=0.01, eps=1.0e-8),
    SGD: dict(lr=0.01),
}

DEVICEID = "npu:0"


class TestSplit(unittest.TestCase):

    def setUp(self):
        self.indices = torch.Tensor([0, 1, 2, 3, 1]).to(torch.int64)
        self.offsets = torch.Tensor([0, 2, 4, 5]).to(torch.int64)
        self.hash_indices = torch.Tensor([0, 1, 2, 3]).to(torch.int64)
        self.unique_indices = torch.Tensor([0, 1, 2, 3]).to(torch.int64)
        self.unique_inverse = torch.Tensor([0, 1, 2, 3, 1]).to(torch.int64)
        self.per_sample_weights = torch.Tensor([1.0, 2.0])
        self.batch_size_per_feature_per_rank = ([1, 1], [1, 1])
        tables = [[100, 32], [200, 64]]
        self.embedding_specs = [
        (num_embeddings, embedding_dim, EmbeddingLocation.HOST, ComputeDevice.CPU)
        for (num_embeddings, embedding_dim) in tables
    ]
 
    @parameterized.expand([
        ("SGD", SGD),
        ("Adagrad", Adagrad),
        ("Adam", Adam),
    ])
    def test_forward_with_all_parameter_return_success(self, name, optim):
        tbe = HybridSplitTableBatchedEmbeddingBagsCodegen(
            self.embedding_specs,
            optimizer=TORCH_OPTIMIZER_TO_FBGEMM[optim],
            pooling_mode=PoolingMode.SUM, 
        )
        result = tbe(self.indices,
            self.offsets,
            self.hash_indices,
            self.unique_indices,
            self.unique_inverse)
        assert(result is not None)
    
    def test_forward_with_unsupported_optim(self):
        tbe = HybridSplitTableBatchedEmbeddingBagsCodegen(
            self.embedding_specs,
            optimizer=EmbOptimType.EXACT_ROWWISE_ADAGRAD,
            pooling_mode=PoolingMode.SUM,
        )
        assert(tbe(self.indices,
            self.offsets,
            self.hash_indices,
            self.unique_indices,
            self.unique_inverse) == NotImplemented)

    

class TestHybridBatchedFusedEmbeddingBag(unittest.TestCase):
    def setUp(self):
        embedding_tables = [ShardedEmbeddingTable(num_embeddings=400, embedding_dim=32, name='table_0', data_type=DataType.FP32, feature_names=["feat_0"])]

        self.config = GroupedEmbeddingConfig(dataType=DataType.FP32, pooling=PoolingType.SUM, is_weighted=False, has_feature_processor=False, compute_kernel="fused", embedding_tables=embedding_tables)
        self.pg = dist.new_group(backend="gloo")
        self.device = torch.device("npu")
        self.sharding_type = ShardingType.ROW_WISE
        
    def test_hybrid_batchedFused_success(self):
        a = HybridBatchedFusedEmbeddingBag(self.config, self.pg, self.device, self.sharding_type)

    