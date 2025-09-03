#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You License at
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
import logging
import sysconfig

import pytest
import fbgemm_gpu
import numpy as np
import torch_npu
import torch

DEVICE = "npu:0"
logging.getLogger().setLevel(logging.INFO)
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

DENSE_DIM0 = [128, 40] # 测试不同batch大小
DENSE_DIM1 = [210] # 固定特征维度1
DENSE_DIM2 = [1, 8] # 固定特征维度2
DIM_LIST = list(itertools.product(DENSE_DIM0, DENSE_DIM1, DENSE_DIM2))

DENSE_DATATYPE = [torch.float32, torch.int64, torch.bfloat16, torch.float16] # 增加BF16和FP16支持
OFFSET_DATATYPE = [torch.int32, torch.int64] # 偏移量数据类型
TYPE_LIST = list(itertools.product(DENSE_DATATYPE, OFFSET_DATATYPE))

# 边界测试用例
EDGE_CASE_DIMS = [
    (1, 10, 1),      # 最小batch和特征维度
    (10, 1, 16),     # 最小序列长度
    (1, 1, 1),       # 所有维度都最小
    (256, 500, 32),  # 较大的batch和特征维度
]

def get_result(device, denses, offsets, types, use_output_size):
    dense_datatype, offset_datatype = types
    dense_torch = torch.from_numpy(denses).to(dense_datatype).to(device)
    offsets_torch = torch.from_numpy(offsets).to(offset_datatype).to(device)

    # 计算累积偏移量
    jagged_id_offset = torch.ops.fbgemm.asynchronous_complete_cumsum(offsets_torch)

    # 获取输出大小（最后一个偏移量即总元素数）
    output_size = None
    if use_output_size:
        output_size = jagged_id_offset[-1]

    # 执行核心操作：稠密张量→不规则张量
    jagged_embedding = torch.ops.fbgemm.dense_to_jagged(dense_torch, [jagged_id_offset], output_size)[0]
    return jagged_embedding.cpu()


@pytest.mark.parametrize("dims", DIM_LIST)
@pytest.mark.parametrize("types", TYPE_LIST)
@pytest.mark.parametrize("use_output_size", [True, False])  # 测试是否传入 output_size
def test_dense_to_jagged(dims, types, use_output_size):
    dense_dim0, dense_dim1, dense_dim2 = dims
    # 1. 生成随机输入数据
    dense_datatype, _ = types
    # 根据目标数据类型生成相应的numpy数据
    if dense_datatype in [torch.bfloat16, torch.float16]:
        denses = np.random.randn(dense_dim0, dense_dim1, dense_dim2).astype(np.float32)
    else:
        denses = np.random.randn(dense_dim0, dense_dim1, dense_dim2).astype(np.float32)
    offsets = np.random.randint(0, dense_dim1, dense_dim0) # 生成随机偏移量

    # 2. 分别获取CPU和NPU结果
    golden_result = get_result(torch.device("cpu"), denses, offsets, types, use_output_size)
    npu_result = get_result(torch.device(DEVICE), denses, offsets, types, use_output_size)

    # 3. 结果比对（允许1e-4的误差，BF16/FP16精度较低，使用1e-3误差）
    tolerance = 1e-3 if types[0] in [torch.bfloat16, torch.float16] else 1e-4
    result_forward = torch.abs(golden_result[0] - npu_result[0]) < tolerance
    logging.info(result_forward.all().item())  # 输出是否全部通过验证


@pytest.mark.parametrize("dims", EDGE_CASE_DIMS)
@pytest.mark.parametrize("dense_dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
def test_dense_to_jagged_edge_cases(dims, dense_dtype, offset_dtype):
    """边界情况测试：测试各种极端维度组合"""
    dense_dim0, dense_dim1, dense_dim2 = dims
    
    # 生成测试数据
    if dense_dtype in [torch.bfloat16, torch.float16]:
        denses = np.random.randn(dense_dim0, dense_dim1, dense_dim2).astype(np.float32)
    else:
        denses = np.random.randn(dense_dim0, dense_dim1, dense_dim2).astype(np.float32)
    offsets = np.random.randint(0, dense_dim1, dense_dim0)
    
    types = (dense_dtype, offset_dtype)
    
    # 获取结果
    golden_result = get_result(torch.device("cpu"), denses, offsets, types, False)
    npu_result = get_result(torch.device(DEVICE), denses, offsets, types, False)
    
    # 结果比对
    tolerance = 1e-3 if dense_dtype in [torch.bfloat16, torch.float16] else 1e-4
    result_forward = torch.abs(golden_result[0] - npu_result[0]) < tolerance
    assert result_forward.all().item(), f"Edge case test failed for dims={dims}, dense_dtype={dense_dtype}, offset_dtype={offset_dtype}"


def test_dense_to_jagged_empty_offsets():
    """测试空偏移量的情况"""
    # 创建空的偏移量
    denses = np.random.randn(1, 10, 8).astype(np.float32)
    offsets = np.array([0, 0])  # 空偏移量
    
    types = (torch.float32, torch.int64)
    
    golden_result = get_result(torch.device("cpu"), denses, offsets, types, False)
    npu_result = get_result(torch.device(DEVICE), denses, offsets, types, False)
    
    tolerance = 1e-4
    result_forward = torch.abs(golden_result[0] - npu_result[0]) < tolerance
    assert result_forward.all().item(), "Empty offsets test failed"


def test_dense_to_jagged_large_offsets():
    """测试大偏移量的情况"""
    # 创建大的偏移量
    denses = np.random.randn(5, 100, 16).astype(np.float32)
    # 创建较大的偏移量，但不超过dense_dim1
    offsets = np.array([0, 20, 40, 60, 80, 100])
    
    types = (torch.float32, torch.int64)
    
    golden_result = get_result(torch.device("cpu"), denses, offsets, types, False)
    npu_result = get_result(torch.device(DEVICE), denses, offsets, types, False)
    
    tolerance = 1e-4
    result_forward = torch.abs(golden_result[0] - npu_result[0]) < tolerance
    assert result_forward.all().item(), "Large offsets test failed"


@pytest.mark.parametrize("dense_dtype", [torch.bfloat16, torch.float16])
def test_bf16_fp16_precision(dense_dtype):
    """专门测试BF16和FP16精度"""
    # 创建特定测试数据
    denses = np.random.randn(10, 50, 8).astype(np.float32)
    offsets = np.random.randint(0, 50, 10)
    
    types = (dense_dtype, torch.int64)
    
    golden_result = get_result(torch.device("cpu"), denses, offsets, types, False)
    npu_result = get_result(torch.device(DEVICE), denses, offsets, types, False)
    
    # BF16/FP16精度较低，使用更大的容差
    tolerance = 1e-2
    result_forward = torch.abs(golden_result[0] - npu_result[0]) < tolerance
    assert result_forward.all().item(), f"BF16/FP16 precision test failed for {dense_dtype}"