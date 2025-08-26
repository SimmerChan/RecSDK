# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Dict, List, Tuple, TypeVar, Generic, Union

import torch

from torchrec.sparse.jagged_tensor import (
    JaggedTensor,
    KeyedJaggedTensor,
)
from torchrec.pt2.checks import  is_non_strict_exporting

T = TypeVar('T', bound='ExtendedJaggedTensor')
K = TypeVar('K', bound='KeyedExtendedJaggedTensor')


class ExtendedJaggedTensor(JaggedTensor):
    """
    Base class for JaggedTensor with an additional tensor field.
    """
    
    def __init__(
        self,
        values: torch.Tensor,
        extra: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        extra_field_name: str = "extra",
    ) -> None:
        if extra is not None and values.size() != extra.size():
            raise ValueError(
                f"{extra_field_name} size must same with values, but got {extra_field_name} size:{extra.size()},"
                f" values size:{values.size()}."
            )

        super().__init__(values, weights, lengths, offsets)
        self._extra = extra
        self._extra_field_name = extra_field_name

    def get_extra(self) -> Optional[torch.Tensor]:
        """Get the extra tensor field."""
        return self._extra


class KeyedExtendedJaggedTensor(KeyedJaggedTensor):
    """
    Base class for KeyedJaggedTensor with an additional tensor field.
    """
    
    def __init__(
        self,
        keys: List[str],
        values: torch.Tensor,
        extra: Optional[torch.Tensor] = None,
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
        extra_field_name: str = "extra",
    ) -> None:
        super().__init__(
            keys,
            values,
            weights,
            lengths,
            offsets,
            stride,
            stride_per_key_per_rank,
            stride_per_key,
            length_per_key,
            lengths_offset_per_key,
            offset_per_key,
            index_per_key,
            jt_dict,
            inverse_indices,
        )

        self._extra: Optional[torch.Tensor] = extra
        self._extra_field_name = extra_field_name

    def get_extra(self) -> Optional[torch.Tensor]:
        """Get the extra tensor field."""
        return self._extra

    @classmethod
    def from_jt_dict_base(
        cls,
        jt_dict: Dict[str, JaggedTensorWithExtra],
        extra_field_name: str = "extra",
    ) -> "KeyedJaggedTensorWithExtra":
        """
        Base implementation for constructing from a dictionary of JaggedTensorWithExtra.
        """
        # 处理空字典的情况
        if not jt_dict:
            return cls(
                keys=[],
                values=torch.empty(0, dtype=torch.int64),
                extra=torch.empty(0, dtype=torch.int64),
                extra_field_name=extra_field_name,
            )
            
        kjt_keys = list(jt_dict.keys())
        kjt_vals_list: List[torch.Tensor] = []
        kjt_extra_list: List[torch.Tensor] = []
        kjt_lens_list: List[torch.Tensor] = []
        kjt_weights_list: List[torch.Tensor] = []
        stride_per_key: List[int] = []
        
        for jt in jt_dict.values():
            stride_per_key.append(len(jt.lengths()))
            kjt_vals_list.append(jt.values())
            kjt_extra_list.append(jt.get_extra())
            kjt_lens_list.append(jt.lengths())
            weight = jt.weights_or_none()
            if weight is not None:
                kjt_weights_list.append(weight)
                
        kjt_vals = torch.concat(kjt_vals_list)
        kjt_lens = torch.concat(kjt_lens_list)

        # handle custom attribute: extra
        kjt_extra = (
            torch.concat(kjt_extra_list) if len(kjt_extra_list) > 0 else None
        )

        kjt_weights = (
            torch.concat(kjt_weights_list) if len(kjt_weights_list) > 0 else None
        )
        kjt_stride, kjt_stride_per_key_per_rank = (
            (stride_per_key[0], None)
            if all(s == stride_per_key[0] for s in stride_per_key)
            else (None, [[stride] for stride in stride_per_key])
        )
        
        kjt = cls(
            keys=kjt_keys,
            values=kjt_vals,
            extra=kjt_extra,
            weights=kjt_weights,
            lengths=kjt_lens,
            stride=kjt_stride,
            stride_per_key_per_rank=kjt_stride_per_key_per_rank,
            extra_field_name=extra_field_name,
        ).sync()
        return kjt

    def split_base(self, segments: List[int], cls_type) -> List["KeyedJaggedTensorWithExtra"]:
        """
        Base implementation for split method.
        """
        split_list: List[KeyedJaggedTensorWithExtra] = []
        start = 0
        start_offset = 0
        _length_per_key = self.length_per_key()
        _offset_per_key = self.offset_per_key()
        
        for segment in segments:
            end = start + segment
            end_offset = _offset_per_key[end]
            keys: List[str] = self._keys[start:end]

            stride, stride_per_key_per_rank = (
                (None, self.stride_per_key_per_rank()[start:end])
                if self.variable_stride_per_key()
                else (self._stride, None)
            )
            
            if segment == len(self._keys):
                # no torch slicing required
                split_list.append(
                    cls_type(
                        keys=self._keys,
                        values=self._values,
                        extra=self._extra,
                        weights=self.weights_or_none(),
                        lengths=self._lengths,
                        offsets=self._offsets,
                        stride=stride,
                        stride_per_key_per_rank=stride_per_key_per_rank,
                        length_per_key=self._length_per_key,
                        offset_per_key=self._offset_per_key,
                        index_per_key=self._index_per_key,
                        jt_dict=self._jt_dict,
                        extra_field_name=self._extra_field_name,
                    )
                )
            elif segment == 0:
                empty_int_list: List[int] = torch.jit.annotate(List[int], [])
                split_list.append(
                    cls_type(
                        keys=keys,
                        values=torch.tensor(
                            empty_int_list,
                            device=self.device(),
                            dtype=self._values.dtype,
                        ),
                        extra=torch.tensor(
                            empty_int_list,
                            device=self.device(),
                            dtype=self._extra.dtype,
                        ),
                        weights=(
                            None
                            if self.weights_or_none() is None
                            else torch.tensor(
                                empty_int_list,
                                device=self.device(),
                                dtype=self.weights().dtype,
                            )
                        ),
                        lengths=torch.tensor(
                            empty_int_list, device=self.device(), dtype=torch.int
                        ),
                        offsets=torch.tensor(
                            empty_int_list, device=self.device(), dtype=torch.int
                        ),
                        stride=stride,
                        stride_per_key_per_rank=stride_per_key_per_rank,
                        length_per_key=None,
                        offset_per_key=None,
                        index_per_key=None,
                        jt_dict=None,
                        extra_field_name=self._extra_field_name,
                    )
                )
            else:
                split_length_per_key = _length_per_key[start:end]
                split_list.append(
                    cls_type(
                        keys=keys,
                        values=self._values[start_offset:end_offset],
                        extra=(
                            self._extra[start_offset:end_offset]
                            if self._extra is not None
                            else None
                        ),
                        weights=(
                            None
                            if self.weights_or_none() is None
                            else self.weights()[start_offset:end_offset]
                        ),
                        lengths=self.lengths()[
                            self.lengths_offset_per_key()[
                                start
                            ]: self.lengths_offset_per_key()[end]
                        ],
                        offsets=None,
                        stride=stride,
                        stride_per_key_per_rank=stride_per_key_per_rank,
                        length_per_key=split_length_per_key,
                        offset_per_key=None,
                        index_per_key=None,
                        jt_dict=None,
                        extra_field_name=self._extra_field_name,
                    )
                )
            start = end
            start_offset = end_offset
        return split_list

    def _validate_permuted_length_per_key_sum(self, permuted_length_per_key_sum: int) -> None:
        """Validate permuted_length_per_key_sum value."""
        if not torch.jit.is_scripting() and is_non_strict_exporting():
            if permuted_length_per_key_sum <= 0:
                raise ValueError("permuted_length_per_key_sum needs to be greater than 0")

    def pin_memory_base(self, cls_type) -> "KeyedJaggedTensorWithExtra":
        """Base implementation for pin_memory method."""
        weights = self._weights
        lengths = self._lengths
        offsets = self._offsets
        stride, stride_per_key_per_rank = (
            (None, self._stride_per_key_per_rank)
            if self.variable_stride_per_key()
            else (self._stride, None)
        )

        return cls_type(
            keys=self._keys,
            values=self._values.pin_memory(),
            extra=(
                self._extra.pin_memory() if self._extra is not None else None
            ),
            weights=weights.pin_memory() if weights is not None else None,
            lengths=lengths.pin_memory() if lengths is not None else None,
            offsets=offsets.pin_memory() if offsets is not None else None,
            stride=stride,
            stride_per_key_per_rank=stride_per_key_per_rank,
            length_per_key=self._length_per_key,
            offset_per_key=self._offset_per_key,
            index_per_key=self._index_per_key,
            jt_dict=None,
            extra_field_name=self._extra_field_name,
        )

    def to_base(self, device: torch.device, non_blocking: bool, cls_type) -> "KeyedJaggedTensorWithExtra":
        """Base implementation for to method."""
        weights = self._weights
        lengths = self._lengths
        offsets = self._offsets
        stride, stride_per_key_per_rank = (
            (None, self._stride_per_key_per_rank)
            if self.variable_stride_per_key()
            else (self._stride, None)
        )
        length_per_key = self._length_per_key
        offset_per_key = self._offset_per_key
        index_per_key = self._index_per_key
        jt_dict = self._jt_dict

        return cls_type(
            keys=self._keys,
            values=self._values.to(device, non_blocking=non_blocking),
            extra=(
                self._extra.to(device, non_blocking=non_blocking)
                if self._extra is not None
                else None
            ),
            weights=(
                weights.to(device, non_blocking=non_blocking)
                if weights is not None
                else None
            ),
            lengths=(
                lengths.to(device, non_blocking=non_blocking)
                if lengths is not None
                else None
            ),
            offsets=(
                offsets.to(device, non_blocking=non_blocking)
                if offsets is not None
                else None
            ),
            stride=stride,
            stride_per_key_per_rank=stride_per_key_per_rank,
            length_per_key=length_per_key,
            offset_per_key=offset_per_key,
            index_per_key=index_per_key,
            jt_dict=jt_dict,
            extra_field_name=self._extra_field_name,
        )

    @torch.jit.unused
    def record_stream_base(self, stream: torch.cuda.streams.Stream) -> None:
        """Base implementation for record_stream method."""
        super().record_stream(stream)
        if self._extra is not None:
            self._extra.record_stream(stream)