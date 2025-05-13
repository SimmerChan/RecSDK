#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, TypeVar, Tuple

import torch
import torch.distributed as dist

from hybrid_torchrec.distributed.embedding_lookup import (
    HybridGroupedPooledEmbeddingsLookup,
)
from hybrid_torchrec.distributed.sharding.post_input_dist import (
    SparseFeaturesPostDist,
    EMPTY_POST_INPUT_DIST,
    UniqueHashFeatureProcess,
    get_feature_len_groupby_table_name,
)
from hybrid_torchrec.modules.hash_embeddingbag import HashMap
from hybrid_torchrec.modules.ids_process import block_bucketize_sparse_features_cpu
from torchrec.distributed.embedding_sharding import (
    BaseEmbeddingLookup,
    BaseSparseFeaturesDist,
    EmbeddingShardingInfo,
)
from torchrec.distributed.embedding_sharding import (
    _fx_wrap_batch_size_per_feature,
    _fx_wrap_max_B,
    _fx_wrap_tensor_to_device_dtype,
    _fx_wrap_gen_list_n_times,
    _fx_wrap_stride,
    _fx_wrap_stride_per_key_per_rank,
)
from torchrec.distributed.embedding_types import (
    BaseGroupedFeatureProcessor,
)
from torchrec.distributed.sharding.rw_sharding import (
    RwPooledEmbeddingSharding,
    RwSparseFeaturesDist,
)
from torchrec.distributed.types import Awaitable
from torchrec.distributed.types import (
    QuantizedCommCodecs,
    ShardingEnv,
    ShardingType,
)
from torchrec.fx.utils import assert_fx_safe
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Multistreamable


class InputDistThreadPoolExecutorSingleton:
    _instance: "InputDistThreadPoolExecutorSingleton" = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(InputDistThreadPoolExecutorSingleton, cls).__new__(
                cls, *args, **kwargs
            )
            default_post_input_threads = 6
            max_threads = (
                default_post_input_threads
                if "INPUT_DIST_THREADS" not in os.environ
                else int(os.environ["INPUT_DIST_THREADS"])
            )
            cls.executor = ThreadPoolExecutor(max_threads)
        return cls._instance


C = TypeVar("C", bound=Multistreamable)
F = TypeVar("F", bound=Multistreamable)
T = TypeVar("T")
W = TypeVar("W")


def bucketize_kjt_before_all2all(
        kjt: KeyedJaggedTensor,
        num_buckets: int,
        block_sizes: torch.Tensor,
        options: Optional[dict] = None,
) -> Tuple[KeyedJaggedTensor, Optional[torch.Tensor]]:
    num_features = len(kjt.keys())
    assert_fx_safe(
        block_sizes.numel() == num_features,
        f"Expecting block sizes for {num_features} features, but {block_sizes.numel()} received.",
    )
    block_sizes_new_type = _fx_wrap_tensor_to_device_dtype(block_sizes, kjt.values())

    # Extract options with default values
    output_permute = options.get('output_permute', False) if options else False
    bucketize_pos = options.get('bucketize_pos', False) if options else False
    block_bucketize_row_pos = options.get('block_bucketize_row_pos', None) if options else None
    keep_original_indices = options.get('keep_original_indices', False) if options else False
    do_unique = options.get('do_unique', False) if options else False

    (
        bucketized_lengths,
        bucketized_indices,
        bucketized_weights,
        pos,
        unbucketize_permute,
        _,
    ) = block_bucketize_sparse_features_cpu(
        kjt.lengths().view(-1),
        kjt.values(),
        bucketize_pos=bucketize_pos,
        sequence=output_permute,
        block_sizes=block_sizes_new_type,
        my_size=num_buckets,
        weights=kjt.weights_or_none(),
        batch_size_per_feature=_fx_wrap_batch_size_per_feature(kjt),
        max_B=_fx_wrap_max_B(kjt),
        block_bucketize_pos=block_bucketize_row_pos,
        keep_orig_idx=keep_original_indices,
        do_unique=do_unique
    )
    return (
        KeyedJaggedTensor(
            keys=_fx_wrap_gen_list_n_times(kjt.keys(), num_buckets),
            values=bucketized_indices,
            weights=pos if bucketize_pos else bucketized_weights,
            lengths=bucketized_lengths.view(-1),
            offsets=None,
            stride=_fx_wrap_stride(kjt),
            stride_per_key_per_rank=_fx_wrap_stride_per_key_per_rank(kjt, num_buckets),
            length_per_key=None,
            offset_per_key=None,
            index_per_key=None,
        ),
        unbucketize_permute,
    )


class HashRwSparseFeaturesDistAwaitable(Awaitable):
    def __init__(self, function, module, sparse_feature: KeyedJaggedTensor, context) -> None:
        super().__init__()
        self.future = InputDistThreadPoolExecutorSingleton().executor.submit(
            function, sparse_feature
        )
        self.pg = module.pg
        self._context = context

    def _wait_impl(self) -> Any:
        result, unbucketize_permute_tensor = self.future.result()
        if (self._context is not None):
            self._context.unbucketize_permute_tensor = unbucketize_permute_tensor
        return result


class HashRwSparseFeaturesDist(RwSparseFeaturesDist):
    def __init__(
        self,
        pg: dist.ProcessGroup,
        num_features: int,
        feature_hash_sizes: List[int],
        feature_total_num_buckets: Optional[List[int]] = None,
        device: Optional[torch.device] = None,
        is_sequence: bool = False,
        has_feature_processor: bool = False,
        need_pos: bool = False,
        keep_original_indices: bool = False,
    ) -> None:
        super().__init__(
            pg,
            num_features,
            feature_hash_sizes,
            feature_total_num_buckets,
            device,
            is_sequence,
            has_feature_processor,
            need_pos,
            keep_original_indices,
        )
        self.pg = pg
        self.do_unique = bool(
                os.environ.get("DO_LOCAL_UNIQUE", False)
            )

    def forward_function(
        self,
        sparse_features: KeyedJaggedTensor,
    ) -> Awaitable[Awaitable[KeyedJaggedTensor]]:
        # 创建一个包含可选参数的字典
        options = {
            'output_permute': self._is_sequence,
            'bucketize_pos': (
                self._has_feature_processor
                if sparse_features.weights_or_none() is None
                else self._need_pos
            ),
            'keep_original_indices': self._keep_original_indices,
            'do_unique': self.do_unique,
        }

        # 调用优化后的函数，将可选参数字典传递给options参数
        (
            bucketized_features,
            unbucketize_permute_tensor,
        ) = bucketize_kjt_before_all2all(
            sparse_features,
            num_buckets=self._world_size,
            block_sizes=self._feature_block_sizes_tensor,
            options=options,
        )
        result = self._dist(bucketized_features)
        if isinstance(result, Awaitable):
            result = result.wait()
        return result, unbucketize_permute_tensor

    def forward(
        self,
        sparse_features: KeyedJaggedTensor,
        context = None
    ) -> Awaitable[Awaitable[KeyedJaggedTensor]]:
        return HashRwSparseFeaturesDistAwaitable(
            self.forward_function, self, sparse_features, context
        )


class HybridRwPooledEmbeddingSharding(RwPooledEmbeddingSharding):

    def __init__(
        self,
        sharding_infos: List[EmbeddingShardingInfo],
        env: ShardingEnv,
        host_env: ShardingEnv,
        device: Optional[torch.device] = None,
        need_pos: bool = False,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        self._host_pg = host_env.process_group
        super().__init__(sharding_infos, env, device, need_pos, qcomm_codecs_registry)

    def create_input_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseSparseFeaturesDist[KeyedJaggedTensor]:
        num_features = self._get_num_features()
        feature_hash_sizes = self._get_feature_hash_sizes()
        return RwSparseFeaturesDist(
            pg=self._host_pg,
            num_features=num_features,
            feature_hash_sizes=feature_hash_sizes,
            device="cpu",
            is_sequence=False,
            has_feature_processor=self._has_feature_processor,
            need_pos=self._need_pos,
        )

    def create_lookup(
        self,
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
        feature_processor: Optional[BaseGroupedFeatureProcessor] = None,
    ) -> BaseEmbeddingLookup:
        return HybridGroupedPooledEmbeddingsLookup(
            grouped_configs=self._grouped_embedding_configs,
            pg=self._pg,
            device=device if device is not None else self._device,
            feature_processor=feature_processor,
            sharding_type=ShardingType.ROW_WISE,
        )

    def create_post_input_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseSparseFeaturesDist[KeyedJaggedTensor]:
        return EMPTY_POST_INPUT_DIST


class HybridHashRwPooledEmbeddingSharding(HybridRwPooledEmbeddingSharding):
    def __init__(
        self,
        sharding_infos: List[EmbeddingShardingInfo],
        table2hashmap: Dict[str, HashMap],
        env: ShardingEnv,
        host_env: ShardingEnv,
        device: Optional[torch.device] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(sharding_infos, env, host_env, device, qcomm_codecs_registry)
        self.table2hashmap = table2hashmap

    def create_input_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseSparseFeaturesDist[KeyedJaggedTensor]:
        num_features = self._get_num_features()
        feature_hash_sizes = self._get_feature_hash_sizes()
        return HashRwSparseFeaturesDist(
            pg=self._host_pg,
            num_features=num_features,
            feature_hash_sizes=feature_hash_sizes,
            device="cpu",
            is_sequence=False,
            has_feature_processor=self._has_feature_processor,
            need_pos=self._need_pos,
        )

    def create_post_input_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseSparseFeaturesDist[KeyedJaggedTensor]:

        table_names, features_split_by_table_name = get_feature_len_groupby_table_name(
            self._grouped_embedding_configs
        )
        hashmaps = [self.table2hashmap[n] for n in table_names]
        feature_processor = UniqueHashFeatureProcess(
            table_names, features_split_by_table_name, hashmaps
        )
        return SparseFeaturesPostDist(feature_processor)
