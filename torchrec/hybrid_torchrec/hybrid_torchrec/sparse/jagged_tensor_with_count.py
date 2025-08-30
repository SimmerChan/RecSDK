#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, Dict, List, Tuple

import torch

from torch.autograd.profiler import record_function
from torchrec.sparse.jagged_tensor import (
    _pin_and_move,
    _permute_tensor_by_segments,
    _sum_by_splits,
    JaggedTensor,
    KeyedJaggedTensor,
)
from torchrec.pt2.checks import is_torchdynamo_compiling, is_non_strict_exporting
from .extended_jagged_tensor import ExtendedJaggedTensor, KeyedExtendedJaggedTensor


class JaggedTensorWithCount(ExtendedJaggedTensor):
    _fields = [
        "_counts"
    ]

    def __init__(
        self,
        values: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        counts: Optional[torch.Tensor] = None,
    ) -> None:
        if counts is not None and values.size() != counts.size():
            raise ValueError(f"counts size must same with values, but got timestamp size:{counts.size()},"
                             f" values size:{values.size()}.")

        super().__init__(
            values=values,
            extra=counts,
            weights=weights,
            lengths=lengths,
            offsets=offsets,
            extra_field_name="counts"
        )
        # values中每个ids出现次数，分桶去重时会进行计算，input_dist all2all会做集合通信，post dist input时做count记录
        self._counts = counts

    @property
    def counts(self):
        return self._counts


class KeyedJaggedTensorWithCount(KeyedExtendedJaggedTensor):
    _fields = [
        "_counts"
    ]

    def __init__(
        self,
        keys: List[str],
        values: torch.Tensor,
        counts: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        stride: Optional[int] = None,
        stride_per_key_per_rank: Optional[List[List[int]]] = None,
        # Below exposed to ensure torch.script-able
        stride_per_key: Optional[List[int]] = None,
        length_per_key: Optional[List[int]] = None,
        lengths_offset_per_key: Optional[List[int]] = None,
        offset_per_key: Optional[List[int]] = None,
        index_per_key: Optional[Dict[str, int]] = None,
        jt_dict: Optional[Dict[str, JaggedTensor]] = None,
        inverse_indices: Optional[Tuple[List[str], torch.Tensor]] = None,
    ) -> None:
        super().__init__(
            keys=keys,
            values=values,
            extra=counts,
            weights=weights,
            lengths=lengths,
            offsets=offsets,
            stride=stride,
            stride_per_key_per_rank=stride_per_key_per_rank,
            stride_per_key=stride_per_key,
            length_per_key=length_per_key,
            lengths_offset_per_key=lengths_offset_per_key,
            offset_per_key=offset_per_key,
            index_per_key=index_per_key,
            jt_dict=jt_dict,
            inverse_indices=inverse_indices,
            extra_field_name="counts"
        )
        self._counts: torch.Tensor = counts

    @property
    def counts(self) -> torch.Tensor:
        return self._counts

    @staticmethod
    def from_jt_dict(jt_dict: Dict[str, JaggedTensorWithCount]) -> "KeyedJaggedTensorWithCount":
        """
        Constructs a KeyedJaggedTensorWithCount from a dictionary of JaggedTensorWithCounts.
        Automatically calls `kjt.sync()` on newly created KJT.

        Args:
            jt_dict (Dict[str, JaggedTensorWithCount]): dictionary of JaggedTensorWithCounts.

        Returns:
            KeyedJaggedTensorWithCount: constructed KeyedJaggedTensorWithCount.
        """
        return KeyedJaggedTensorWithCount.from_jt_dict_base(jt_dict, "counts")

    def split(self, segments: List[int]) -> List["KeyedJaggedTensorWithCount"]:
        return self.split_base(segments, KeyedJaggedTensorWithCount)

    def permute(
        self, indices: List[int], indices_tensor: Optional[torch.Tensor] = None
    ) -> "KeyedJaggedTensorWithCount":
        """
        Permutes the KeyedJaggedTensorWithCount.

        Args:
            indices (List[int]): list of indices.
            indices_tensor (Optional[torch.Tensor]): tensor of indices.

        Returns:
            KeyedJaggedTensorWithCount: permuted KeyedJaggedTensorWithCount.
        """
        if indices_tensor is None:
            indices_tensor = torch.tensor(
                indices, dtype=torch.int, device=self.device()
            )

        length_per_key = self.length_per_key()
        permuted_keys: List[str] = []
        permuted_stride_per_key_per_rank: List[List[int]] = []
        permuted_length_per_key: List[int] = []
        permuted_length_per_key_sum = 0
        for index in indices:
            key = self.keys()[index]
            permuted_keys.append(key)
            permuted_length_per_key.append(length_per_key[index])
            if self.variable_stride_per_key():
                permuted_stride_per_key_per_rank.append(
                    self.stride_per_key_per_rank()[index]
                )

        permuted_length_per_key_sum = sum(permuted_length_per_key)
        # 避免直接访问受保护的成员
        if not torch.jit.is_scripting() and is_non_strict_exporting():
            torch._check_is_size(permuted_length_per_key_sum)
            torch._check(permuted_length_per_key_sum != -1)
            torch._check(permuted_length_per_key_sum != 0)

        if self.variable_stride_per_key():
            length_per_key_tensor = _pin_and_move(
                torch.tensor(self.length_per_key()), self.device()
            )
            stride_per_key_tensor = _pin_and_move(
                torch.tensor(self.stride_per_key()), self.device()
            )
            permuted_lengths, _ = _permute_tensor_by_segments(
                self.lengths(),
                stride_per_key_tensor,
                indices_tensor,
                None,
            )
            permuted_values, permuted_weights = _permute_tensor_by_segments(
                self.values(),
                length_per_key_tensor,
                indices_tensor,
                self.weights_or_none(),
            )
            permuted_counts, _ = _permute_tensor_by_segments(
                self.counts,
                length_per_key_tensor,
                indices_tensor,
                self.weights_or_none(),
            )
        elif is_torchdynamo_compiling() and not torch.jit.is_scripting():
            (
                permuted_lengths,
                permuted_values,
                permuted_weights,
            ) = torch.ops.fbgemm.permute_2D_sparse_data_input1D(
                indices_tensor,
                self.lengths(),
                self.values(),
                self.stride(),
                self.weights_or_none(),
                permuted_length_per_key_sum,
            )
            _, permuted_counts, _ = torch.ops.fbgemm.permute_2D_sparse_data_input1D(
                indices_tensor,
                self.lengths(),
                self.counts,
                self.stride(),
                self.weights_or_none(),
                permuted_length_per_key_sum,
            )
        else:
            (
                permuted_lengths,
                permuted_values,
                permuted_weights,
            ) = torch.ops.fbgemm.permute_2D_sparse_data(
                indices_tensor,
                self.lengths().view(len(self._keys), -1),
                self.values(),
                self.weights_or_none(),
                permuted_length_per_key_sum,
            )
            _, permuted_counts, _ = torch.ops.fbgemm.permute_2D_sparse_data(
                indices_tensor,
                self.lengths().view(len(self._keys), -1),
                self.counts,
                self.weights_or_none(),
                permuted_length_per_key_sum,
            )
        stride_per_key_per_rank = (
            permuted_stride_per_key_per_rank if self.variable_stride_per_key() else None
        )
        kjt = KeyedJaggedTensorWithCount(
            keys=permuted_keys,
            values=permuted_values,
            counts=permuted_counts,
            weights=permuted_weights,
            lengths=permuted_lengths.view(-1),
            offsets=None,
            stride=self._stride,
            stride_per_key_per_rank=stride_per_key_per_rank,
            stride_per_key=None,
            length_per_key=permuted_length_per_key if len(permuted_keys) > 0 else None,
            lengths_offset_per_key=None,
            offset_per_key=None,
            index_per_key=None,
            jt_dict=None,
            inverse_indices=None,
        )
        return kjt

    def pin_memory(self) -> "KeyedJaggedTensorWithCount":
        return self.pin_memory_base(KeyedJaggedTensorWithCount)

    def to(
        self, device: torch.device, non_blocking: bool = False
    ) -> "KeyedJaggedTensorWithCount":
        return self.to_base(device, non_blocking, KeyedJaggedTensorWithCount)

    @torch.jit.unused
    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        self.record_stream_base(stream)