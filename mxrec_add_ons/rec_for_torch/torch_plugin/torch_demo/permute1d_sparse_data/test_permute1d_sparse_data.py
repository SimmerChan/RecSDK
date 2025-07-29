#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import itertools
import sysconfig

import pytest
import torch
import torch_npu
import fbgemm_gpu
import numpy as np

# 定义用到的卡以及so库的位置
DEVICE = "npu:0"
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

# 定义参数数据类型
PERMUTE_TYPE = [np.int32]
LENGTHS_TYPE = [np.int64, np.int32]
VALUES_TYPE = [np.int64, np.int32, np.float32]
WEIGHTS_TYPE = [None, np.float32]                    # weights为可选参数
TYPE_LIST = list(itertools.product(PERMUTE_TYPE, LENGTHS_TYPE, VALUES_TYPE, WEIGHTS_TYPE))

# 定义参数shape
# permute, lengths shape为[T]
# extra_t用于测试permute和lengths不等长的情况，lengths[T + extra_T]
T = np.random.randint(2, 30, 4)       # 随机生成4个介于2到30之间的整数，代表稀疏数据的原始维度
EXTRA_T = [0, 3, 8]
SHAPE_LIST = list(itertools.product(T, EXTRA_T))


def get_result(tensors: dict, device: str = 'cpu'):
    tensors = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in tensors.items()}

    # 根据device类型进行npu转换
    if device and device.startswith('npu'):
        torch.npu.set_device(device)
        tensors = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in tensors.items()}

    results = torch.ops.fbgemm.permute_1D_sparse_data(**tensors)
    return [x.cpu() if isinstance(x, torch.Tensor) else x for x in results]


@pytest.mark.parametrize("types", TYPE_LIST)
@pytest.mark.parametrize("shapes", SHAPE_LIST)
@pytest.mark.parametrize("enable_permuted_sum", [True, False])
def test_permute1d_sparse_data(types, shapes, enable_permuted_sum):
    """
    测试正常情况下的permute1d_sparse_data算子功能
    Params:
        permute: (T) dtype=int32
        lengths: (T + T') dtype=ltype
                L = lengths[:T].sum()
        values: (L) dtype=vtype
        weights: (L) dtype=fp32
        permuted_lengths_sum: int
    """
    ptype, ltype, vtype, wtype = types
    t, extra_t = shapes

    permute = np.arange(t, dtype=ptype)  # 创建一个从0到t-1的数组
    np.random.shuffle(permute)  # 随机打乱permute数组
    lengths = np.ones(t + extra_t, dtype=ltype) # 创建全1的lengths，简化
    values = np.arange(0, t + extra_t, dtype=vtype)  # 注意总长度为lengths.sum(),此处特殊，为len(lengths)
    weights = np.arange(0, t + extra_t, dtype=wtype) if wtype else None
    permuted_lengths_sum = t if enable_permuted_sum else None

    params = {
        'permute': permute,
        'lengths': lengths,
        'values': values,
        'weights': weights,
        'permuted_lengths_sum': permuted_lengths_sum
    }

    golden = get_result(params)
    result = get_result(params, DEVICE)

    for gt, pred in zip(golden, result):
        assert type(gt) is type(pred)
        if isinstance(gt, torch.Tensor) and isinstance(pred, torch.Tensor):
            assert torch.allclose(gt, pred, atol=1e-5)


@pytest.mark.parametrize("types", TYPE_LIST)
@pytest.mark.parametrize("shapes", SHAPE_LIST)
def test_longer_permute(types, shapes):
    """
    测试permute长度大于lengths的情况，应该抛出异常
    """
    ptype, ltype, vtype, wtype = types
    t, extra_t = shapes

    # 显式检查输入有效性
    if extra_t <= 0:
        return

    permute = np.arange(t + extra_t, dtype=ptype)
    np.random.shuffle(permute)
    lengths = np.ones(t, dtype=ltype)
    values = np.arange(0, t, dtype=vtype)
    weights = np.arange(0, t, dtype=wtype) if wtype else None

    params = {
        'permute': permute,
        'lengths': lengths,
        'values': values,
        'weights': weights,
        'permuted_lengths_sum': None
    }

    with pytest.raises(RuntimeError):
        result = get_result(params, DEVICE)
        assert result is not None


def test_empty_input():
    """
    测试空输入的情况
    """
    params = {
        'permute': np.array([], dtype=np.int32),
        'lengths': np.array([], dtype=np.int32),
        'values': np.array([], dtype=np.int32),
        'weights': None,
        'permuted_lengths_sum': None
    }

    with pytest.raises(RuntimeError):
        get_result(params, DEVICE)


def test_invalid_weights_length():
    """
    测试weights长度与values不匹配的情况
    """
    t = 5
    params = {
        'permute': np.arange(t, dtype=np.int32),
        'lengths': np.ones(t, dtype=np.int32),
        'values': np.arange(t, dtype=np.int32),
        'weights': np.arange(t + 1, dtype=np.float32),  # 长度不匹配
        'permuted_lengths_sum': None
    }

    with pytest.raises(RuntimeError):
        get_result(params, DEVICE)


def test_2d_input():
    """
    测试输入为2D的情况(应该报错)
    """
    t = 5
    params = {
        'permute': np.arange(t, dtype=np.int32).reshape(1, -1),  # 2D permute
        'lengths': np.ones(t, dtype=np.int32),
        'values': np.arange(t, dtype=np.int32),
        'weights': None,
        'permuted_lengths_sum': None
    }

    with pytest.raises(RuntimeError):
        get_result(params, DEVICE)


def test_negative_lengths():
    """
    测试lengths包含负数的情况
    """
    t = 5
    params = {
        'permute': np.arange(t, dtype=np.int32),
        'lengths': np.array([1, -2, 3, 4, 5], dtype=np.int32),  # 包含负数
        'values': np.arange(15, dtype=np.int32),  # 假设sum(lengths)=11
        'weights': None,
        'permuted_lengths_sum': None
    }

    with pytest.raises(RuntimeError):
        get_result(params, DEVICE)


def test_large_permuted_lengths_sum():
    """
    测试permuted_lengths_sum大于实际长度的情况
    """
    t = 5
    params = {
        'permute': np.arange(t, dtype=np.int32),
        'lengths': np.ones(t, dtype=np.int32),
        'values': np.arange(t, dtype=np.int32),
        'weights': None,
        'permuted_lengths_sum': t + 10  # 大于实际长度
    }

    # 根据实现逻辑，这个测试可能通过或失败，取决于实现方式
    try:
        result = get_result(params, DEVICE)
        assert len(result[1]) == t  # 检查输出values长度
    except RuntimeError:
        pass  # 也允许抛出异常


def test_duplicate_permute_indices():
    """
    测试permute包含重复索引的情况
    """
    t = 5
    params = {
        'permute': np.array([0, 1, 1, 3, 4], dtype=np.int32),  # 包含重复索引
        'lengths': np.ones(t, dtype=np.int32),
        'values': np.arange(t, dtype=np.int32),
        'weights': None,
        'permuted_lengths_sum': None
    }

    # 根据业务需求决定是否允许重复索引
    try:
        result = get_result(params, DEVICE)
        assert len(result[0]) == t  # 检查输出lengths长度
    except RuntimeError:
        pass  # 如果不允许重复索引，则抛出异常


def test_out_of_bound_permute_indices():
    """
    测试permute包含越界索引的情况
    """
    t = 5
    params = {
        'permute': np.array([0, 1, 2, 5, 4], dtype=np.int32),  # 包含越界索引
        'lengths': np.ones(t, dtype=np.int32),
        'values': np.arange(t, dtype=np.int32),
        'weights': None,
        'permuted_lengths_sum': None
    }

    with pytest.raises(RuntimeError):
        get_result(params, DEVICE)


def test_very_large_input():
    """
    测试非常大的输入情况
    """
    t = 10000  # 大尺寸
    permute = np.arange(t, dtype=np.int32)
    np.random.shuffle(permute)
    lengths = np.random.randint(1, 10, size=t, dtype=np.int32)
    total_length = lengths.sum()
    values = np.arange(total_length, dtype=np.int32)
    weights = np.random.rand(total_length).astype(np.float32)

    params = {
        'permute': permute,
        'lengths': lengths,
        'values': values,
        'weights': weights,
        'permuted_lengths_sum': None
    }

    golden = get_result(params)
    result = get_result(params, DEVICE)

    for gt, pred in zip(golden, result):
        assert type(gt) is type(pred)
        if isinstance(gt, torch.Tensor) and isinstance(pred, torch.Tensor):
            assert torch.allclose(gt, pred, atol=1e-5)