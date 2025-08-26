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
        jt_dict: Dict[str, ExtendedJaggedTensor],
        extra_field_name: str = "extra",
    ) -> "KeyedExtendedJaggedTensor":
        """
        Base implementation for constructing from a dictionary of JaggedTensorWithExtra.
        """
        # 处理空字典的情况
        if not jt_dict:
            # 根据extra_field_name动态构造参数字典
            constructor_kwargs = {
                "keys": [],
                "values": torch.empty(0, dtype=torch.int64),
            }
            
            # 根据extra_field_name设置对应的参数
            if extra_field_name == "timestamps":
                constructor_kwargs["timestamps"] = torch.empty(0, dtype=torch.int64)
            elif extra_field_name == "counts":
                constructor_kwargs["counts"] = torch.empty(0, dtype=torch.int64)
            else:
                # 对于其他情况，使用通用的extra参数
                constructor_kwargs["extra"] = torch.empty(0, dtype=torch.int64)
                constructor_kwargs["extra_field_name"] = extra_field_name
            
            return cls(**constructor_kwargs)
            
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
        
        # 根据extra_field_name动态构造参数字典
        constructor_kwargs = {
            "keys": kjt_keys,
            "values": kjt_vals,
            "weights": kjt_weights,
            "lengths": kjt_lens,
            "stride": kjt_stride,
            "stride_per_key_per_rank": kjt_stride_per_key_per_rank,
        }
        
        # 根据extra_field_name设置对应的参数
        if extra_field_name == "timestamps":
            constructor_kwargs["timestamps"] = kjt_extra
        elif extra_field_name == "counts":
            constructor_kwargs["counts"] = kjt_extra
        else:
            # 对于其他情况，使用通用的extra参数
            constructor_kwargs["extra"] = kjt_extra
            constructor_kwargs["extra_field_name"] = extra_field_name
        
        kjt = cls(**constructor_kwargs).sync()
        return kjt

    def split_base(self, segments: List[int], cls_type) -> List["KeyedExtendedJaggedTensor"]:
        """
        Base implementation for split method.
        """
        split_list: List[KeyedExtendedJaggedTensor] = []
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
                # 根据extra_field_name动态构造参数字典
                constructor_kwargs = {
                    "keys": self._keys,
                    "values": self._values,
                    "weights": self.weights_or_none(),
                    "lengths": self._lengths,
                    "offsets": self._offsets,
                    "stride": stride,
                    "stride_per_key_per_rank": stride_per_key_per_rank,
                    "length_per_key": self._length_per_key,
                    "offset_per_key": self._offset_per_key,
                    "index_per_key": self._index_per_key,
                    "jt_dict": self._jt_dict,
                }
                
                # 根据extra_field_name设置对应的参数
                if self._extra_field_name == "timestamps":
                    constructor_kwargs["timestamps"] = self._extra
                elif self._extra_field_name == "counts":
                    constructor_kwargs["counts"] = self._extra
                else:
                    # 对于其他情况，使用通用的extra参数
                    constructor_kwargs["extra"] = self._extra
                    constructor_kwargs["extra_field_name"] = self._extra_field_name
                
                split_list.append(cls_type(**constructor_kwargs))
            elif segment == 0:
                # 根据extra_field_name动态构造参数字典
                empty_int_list: List[int] = torch.jit.annotate(List[int], [])
                constructor_kwargs = {
                    "keys": keys,
                    "values": torch.tensor(
                        empty_int_list,
                        device=self.device(),
                        dtype=self._values.dtype,
                    ),
                    "weights": (
                        None
                        if self.weights_or_none() is None
                        else torch.tensor(
                            empty_int_list,
                            device=self.device(),
                            dtype=self.weights().dtype,
                        )
                    ),
                    "lengths": torch.tensor(
                        empty_int_list, device=self.device(), dtype=torch.int
                    ),
                    "offsets": torch.tensor(
                        empty_int_list, device=self.device(), dtype=torch.int
                    ),
                    "stride": stride,
                    "stride_per_key_per_rank": stride_per_key_per_rank,
                    "length_per_key": None,
                    "offset_per_key": None,
                    "index_per_key": None,
                    "jt_dict": None,
                }
                
                # 根据extra_field_name设置对应的参数
                if self._extra_field_name == "timestamps":
                    constructor_kwargs["timestamps"] = torch.tensor(
                        empty_int_list,
                        device=self.device(),
                        dtype=self._extra.dtype if self._extra is not None else torch.int64,
                    )
                elif self._extra_field_name == "counts":
                    constructor_kwargs["counts"] = torch.tensor(
                        empty_int_list,
                        device=self.device(),
                        dtype=self._extra.dtype if self._extra is not None else torch.int64,
                    )
                else:
                    # 对于其他情况，使用通用的extra参数
                    constructor_kwargs["extra"] = torch.tensor(
                        empty_int_list,
                        device=self.device(),
                        dtype=self._extra.dtype if self._extra is not None else torch.int64,
                    )
                    constructor_kwargs["extra_field_name"] = self._extra_field_name
                
                split_list.append(cls_type(**constructor_kwargs))
            else:
                # 根据extra_field_name动态构造参数字典
                split_length_per_key = _length_per_key[start:end]
                constructor_kwargs = {
                    "keys": keys,
                    "values": self._values[start_offset:end_offset],
                    "weights": (
                        None
                        if self.weights_or_none() is None
                        else self.weights()[start_offset:end_offset]
                    ),
                    "lengths": self.lengths()[
                        self.lengths_offset_per_key()[
                            start
                        ]: self.lengths_offset_per_key()[end]
                    ],
                    "offsets": None,
                    "stride": stride,
                    "stride_per_key_per_rank": stride_per_key_per_rank,
                    "length_per_key": split_length_per_key,
                    "offset_per_key": None,
                    "index_per_key": None,
                    "jt_dict": None,
                }
                
                # 根据extra_field_name设置对应的参数
                if self._extra_field_name == "timestamps":
                    constructor_kwargs["timestamps"] = (
                        self._extra[start_offset:end_offset]
                        if self._extra is not None
                        else None
                    )
                elif self._extra_field_name == "counts":
                    constructor_kwargs["counts"] = (
                        self._extra[start_offset:end_offset]
                        if self._extra is not None
                        else None
                    )
                else:
                    # 对于其他情况，使用通用的extra参数
                    constructor_kwargs["extra"] = (
                        self._extra[start_offset:end_offset]
                        if self._extra is not None
                        else None
                    )
                    constructor_kwargs["extra_field_name"] = self._extra_field_name
                
                split_list.append(cls_type(**constructor_kwargs))
            start = end
            start_offset = end_offset
        return split_list

    def _validate_permuted_length_per_key_sum(self, permuted_length_per_key_sum: int) -> None:
        """Validate permuted_length_per_key_sum value."""
        if not torch.jit.is_scripting() and is_non_strict_exporting():
            if permuted_length_per_key_sum <= 0:
                raise ValueError("permuted_length_per_key_sum needs to be greater than 0")

    def pin_memory_base(self, cls_type) -> "KeyedExtendedJaggedTensor":
        """Base implementation for pin_memory method."""
        weights = self._weights
        lengths = self._lengths
        offsets = self._offsets
        stride, stride_per_key_per_rank = (
            (None, self._stride_per_key_per_rank)
            if self.variable_stride_per_key()
            else (self._stride, None)
        )

        # 根据extra_field_name动态构造参数字典
        constructor_kwargs = {
            "keys": self._keys,
            "values": self._values.pin_memory(),
            "weights": weights.pin_memory() if weights is not None else None,
            "lengths": lengths.pin_memory() if lengths is not None else None,
            "offsets": offsets.pin_memory() if offsets is not None else None,
            "stride": stride,
            "stride_per_key_per_rank": stride_per_key_per_rank,
            "length_per_key": self._length_per_key,
            "offset_per_key": self._offset_per_key,
            "index_per_key": self._index_per_key,
            "jt_dict": None,
        }
        
        # 根据extra_field_name设置对应的参数
        if self._extra_field_name == "timestamps":
            constructor_kwargs["timestamps"] = (
                self._extra.pin_memory() if self._extra is not None else None
            )
        elif self._extra_field_name == "counts":
            constructor_kwargs["counts"] = (
                self._extra.pin_memory() if self._extra is not None else None
            )
        else:
            # 对于其他情况，使用通用的extra参数
            constructor_kwargs["extra"] = (
                self._extra.pin_memory() if self._extra is not None else None
            )
            constructor_kwargs["extra_field_name"] = self._extra_field_name
        
        return cls_type(**constructor_kwargs)

    def to_base(self, device: torch.device, non_blocking: bool, cls_type) -> "KeyedExtendedJaggedTensor":
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

        # 根据extra_field_name动态构造参数字典
        constructor_kwargs = {
            "keys": self._keys,
            "values": self._values.to(device, non_blocking=non_blocking),
            "weights": (
                weights.to(device, non_blocking=non_blocking)
                if weights is not None
                else None
            ),
            "lengths": (
                lengths.to(device, non_blocking=non_blocking)
                if lengths is not None
                else None
            ),
            "offsets": (
                offsets.to(device, non_blocking=non_blocking)
                if offsets is not None
                else None
            ),
            "stride": stride,
            "stride_per_key_per_rank": stride_per_key_per_rank,
            "length_per_key": length_per_key,
            "offset_per_key": offset_per_key,
            "index_per_key": index_per_key,
            "jt_dict": jt_dict,
        }
        
        # 根据extra_field_name设置对应的参数
        if self._extra_field_name == "timestamps":
            constructor_kwargs["timestamps"] = (
                self._extra.to(device, non_blocking=non_blocking)
                if self._extra is not None
                else None
            )
        elif self._extra_field_name == "counts":
            constructor_kwargs["counts"] = (
                self._extra.to(device, non_blocking=non_blocking)
                if self._extra is not None
                else None
            )
        else:
            # 对于其他情况，使用通用的extra参数
            constructor_kwargs["extra"] = (
                self._extra.to(device, non_blocking=non_blocking)
                if self._extra is not None
                else None
            )
            constructor_kwargs["extra_field_name"] = self._extra_field_name
        
        return cls_type(**constructor_kwargs)

    @torch.jit.unused
    def record_stream_base(self, stream: torch.cuda.streams.Stream) -> None:
        """Base implementation for record_stream method."""
        super().record_stream(stream)
        if self._extra is not None:
            self._extra.record_stream(stream)