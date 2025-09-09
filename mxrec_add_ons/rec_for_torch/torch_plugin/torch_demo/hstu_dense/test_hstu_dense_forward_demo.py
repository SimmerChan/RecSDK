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
from ast import main
from copy import deepcopy
import dataclasses
import os
import sysconfig
from enum import Enum

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from test_target_mask import ScoreShapeParam, compute_target_mask_each_block_concat

torch.npu.config.allow_internal_format = False

torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

device_id: int = 0
torch.npu.set_device(device_id)
BLOCK_HEIGHT: int = 256
MAX_NUM_TARGET: int = 512


class MaskType(int, Enum):
    TRIL = 0  # 下三角掩码
    TRIU = 1  # 上三角掩码
    NONE = 2  # 无掩码
    CUSTOM = 3  # 自定义掩码


@dataclasses.dataclass
class MaskParams:
    type: int = MaskType.NONE
    num_context: int = 0
    num_target: int = 0
    target_group_size: int = 0

    def values(self):
        return self.type, self.num_context, self.num_target, self.target_group_size


@dataclasses.dataclass
class QKVShapeParams:
    dtype: torch.dtype
    batch_size: int
    min_seq_len: int = 1
    max_seq_len: int
    num_heads: int
    attention_dim: int

    def values(self):
        return self.batch_size, self.min_seq_len, self.max_seq_len, self.num_heads, self.attention_dim


def get_chip():
    return False


def skip_seq_len(seq_len):
    block_len = 128
    if get_chip() and seq_len % block_len:
        return True
    return False


def cached_create_causal_mask(param: ScoreShapeParam) -> torch.Tensor:
    cached_file = f"cached_target_mask{param.target_group_size}.pt"
    if os.path.exists(cached_file):
        mask = torch.tril(torch.ones(param.seq_len, param.seq_len))
        mask[:param.num_context, :param.num_target] = 1
        if param.num_target > 0:
            target_mask = torch.load(cached_file)
            mask[-param.num_target:, -param.num_target:] = target_mask[:param.num_target, :param.num_target]
        return mask
    else:
        _param = deepcopy(param)
        _param.num_target = MAX_NUM_TARGET
        _param.seq_len += MAX_NUM_TARGET
        mask = compute_target_mask_each_block_concat(_param, use_npu=False)
        torch.save(mask[-MAX_NUM_TARGET:, -MAX_NUM_TARGET:], cached_file)
        return mask[:param.seq_len, :param.seq_len]


def jagged_data_gen(qkv_shape_params: QKVShapeParams, mask_params: MaskParams):
    batch_size, min_seq_len, max_seq_len, num_heads, attention_dim = qkv_shape_params.values()
    mask_type, num_context, num_target, target_group_size = mask_params.values()
    seq_lens = np.random.randint(min_seq_len, max_seq_len + 1, batch_size)
    if mask_type is MaskType.TRIL:
        seq_lens += num_context + num_target

    seq_offset = torch.concat((torch.zeros((1,), dtype=torch.int64),
                               torch.cumsum(torch.from_numpy(seq_lens), axis=0))).to(torch.int64).numpy()

    max_seq_len = np.max(seq_lens.tolist())
    total_seqs = np.sum(seq_lens.tolist())

    q = torch.rand(total_seqs, num_heads, attention_dim).to(torch.float32)
    q = q.uniform_(-1, 1)
    k = torch.rand(total_seqs, num_heads, attention_dim).to(torch.float32)
    k = k.uniform_(-1, 1)
    v = torch.rand(total_seqs, num_heads, attention_dim).to(torch.float32)
    v = v.uniform_(-1, 1)

    rel_attn_bias = torch.zeros(batch_size, num_heads, max_seq_len, max_seq_len).to(torch.float32)
    for batch_id in range(batch_size):
        seq_len = seq_lens[batch_id]
        rel_attn_bias[batch_id, :, 0:seq_len, 0:seq_len] = torch.rand(seq_len, seq_len).to(torch.float32)

    if mask_type == MaskType.TRIL:
        invalid_attn_mask = torch.zeros(batch_size, num_heads, max_seq_len, max_seq_len)
        for sample_id, seq_len in enumerate(seq_lens):
            score_shape_param = ScoreShapeParam(
                seq_len=seq_len,
                num_target=num_target,
                num_context=num_context,
                num_history=seq_len - num_target,
                target_group_size=target_group_size,
                block_h=BLOCK_HEIGHT,
                block_w=BLOCK_HEIGHT
            )
            mask = cached_create_causal_mask(score_shape_param)
            invalid_attn_mask[sample_id, :, :seq_len, :seq_len] = mask
    else:
        invalid_attn_mask = torch.randint(0, 2, size=(batch_size, num_heads, max_seq_len, max_seq_len))
    invalid_attn_mask = invalid_attn_mask.cpu().to(torch.float32)

    return q, k, v, seq_offset, rel_attn_bias, invalid_attn_mask, max_seq_len


def generate_tensor(batch_size, max_seq_len, num_heads, attention_dim, data_type, mask_type):
    total_num = batch_size * max_seq_len * num_heads * attention_dim

    q = torch.rand(total_num).reshape(batch_size, max_seq_len, num_heads, attention_dim)
    k = torch.rand(total_num).reshape(batch_size, max_seq_len, num_heads, attention_dim)
    v = torch.rand(total_num).reshape(batch_size, max_seq_len, num_heads, attention_dim)
    rel_attn_bias = torch.rand(batch_size, num_heads, max_seq_len, max_seq_len)
    if get_chip():
        invalid_attn_mask = torch.randint(0, 2, (max_seq_len, max_seq_len))
        invalid_attn_mask = torch.tril(invalid_attn_mask)
        invalid_attn_mask = invalid_attn_mask.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1)
    elif mask_type == MaskType.TRIL:
        invalid_attn_mask = 1 - torch.triu(torch.ones(batch_size, num_heads, max_seq_len, max_seq_len), diagonal=1)
    else:
        invalid_attn_mask = torch.randint(0, 2, size=(batch_size, num_heads, max_seq_len, max_seq_len))
    return q.to(data_type).to(f"npu:{device_id}"), k.to(data_type).to(f"npu:{device_id}"), v.to(data_type).to(
        f"npu:{device_id}"), rel_attn_bias.to(data_type).to(f"npu:{device_id}"), invalid_attn_mask.to(data_type).to(
        f"npu:{device_id}")


class TestHstuJaggedDemo:
    @staticmethod
    def jagged_to_dense(jagged_tensor, seq_lens, head_nums, atten_dim):
        need_pad_seq = []
        offset = 0
        for seq_len in seq_lens:
            src_tensor = jagged_tensor[offset: offset + seq_len, :, :].reshape(seq_len, head_nums, atten_dim)
            need_pad_seq.append(src_tensor)
            offset = offset + seq_len

        dense_tensor = torch.nn.utils.rnn.pad_sequence(need_pad_seq, batch_first=True)
        return dense_tensor

    @staticmethod
    def dense_to_jagged(q, dense_tensor, seq_lens):
        tensor = torch.zeros_like(q).cpu()

        offset = 0
        for batch_id, seq_len in enumerate(seq_lens):
            tensor[offset: offset + seq_len, :, :] = dense_tensor[batch_id, 0: seq_len, :, :]
            offset = offset + seq_len

        return tensor

    @staticmethod
    def custom_op_exec(q, k, v, seq_offset, bias, mask, max_seq_len, enable_bias, mask_type, silu_scale,
                       data_type, num_context=0, num_target=0, target_group_size=0):
        q_npu = q.to(f"npu:{device_id}").to(data_type)
        k_npu = k.to(f"npu:{device_id}").to(data_type)
        v_npu = v.to(f"npu:{device_id}").to(data_type)
        bias_npu = bias.to(f"npu:{device_id}").to(data_type)
        mask_npu = mask.to(f"npu:{device_id}").to(data_type)

        if enable_bias:
            output = torch.ops.mxrec.hstu_dense(
                q_npu, k_npu, v_npu, mask_npu, bias_npu, mask_type, max_seq_len, silu_scale, "jagged", seq_offset, 
                num_context, num_target, target_group_size
            )
        else:
            output = torch.ops.mxrec.hstu_dense(
                q_npu, k_npu, v_npu, mask_npu, None, mask_type, max_seq_len, silu_scale, "jagged", seq_offset,
                num_context, num_target, target_group_size
            )
        torch.npu.synchronize()
        return output.cpu().to(data_type).reshape(-1)

    def golden_op_exec(self, q, k, v, seq_offset, bias, mask, max_seq_len, enable_bias, mask_type, silu_scale,
                       data_type):
        head_nums = q.shape[1]
        head_dim = q.shape[2]
        batch_size = bias.shape[0]

        seq_lens = np.zeros((batch_size,)).astype(np.int64)
        for batch_id in range(batch_size):
            seq_lens[batch_id] = seq_offset[batch_id + 1] - seq_offset[batch_id]

        silu_scale = 1 / max_seq_len if silu_scale == 0 else silu_scale

        q_dens = self.jagged_to_dense(q, seq_lens, head_nums, head_dim).to(data_type).to(f"npu:{device_id}")
        k_dens = self.jagged_to_dense(k, seq_lens, head_nums, head_dim).to(data_type).to(f"npu:{device_id}")
        v_dens = self.jagged_to_dense(v, seq_lens, head_nums, head_dim).to(data_type).to(f"npu:{device_id}")
        mask = mask.reshape(batch_size, head_nums, max_seq_len, max_seq_len).to(data_type).to(f"npu:{device_id}")
        attn_bias = bias.reshape(batch_size, head_nums, max_seq_len, max_seq_len).to(data_type).to(f"npu:{device_id}")

        q_dens = q_dens.permute(0, 2, 1, 3)
        k_dens = k_dens.permute(0, 2, 3, 1)
        qk_attn = torch.matmul(q_dens, k_dens)

        qk_attn = qk_attn.to(torch.float32)
        attn_bias = attn_bias.to(torch.float32)
        mask = mask.to(torch.float32)
        if enable_bias:
            qk_attn = qk_attn + attn_bias

        qk_attn = F.silu(qk_attn) * silu_scale

        if mask_type != MaskType.NONE:
            qk_attn = qk_attn * mask

        v_dens = v_dens.permute(0, 2, 1, 3)

        qk_attn = qk_attn.to(data_type)
        attn_output = torch.matmul(qk_attn, v_dens)
        attn_output = attn_output.permute(0, 2, 1, 3).cpu()
        attn_output = self.dense_to_jagged(q, attn_output, seq_lens)

        torch.npu.synchronize()
        return attn_output.to(data_type).reshape(-1)

    def execute(self, qkv_shape_params: QKVShapeParams, mask_params: MaskParams, enable_bias: bool, silu_scale: float):
        q, k, v, seq_offset, bias, mask, max_seq_len = jagged_data_gen(qkv_shape_params, mask_params)

        output = self.custom_op_exec(q, k, v, seq_offset, bias, mask, max_seq_len, enable_bias,
                                     mask_params.type, silu_scale, qkv_shape_params.dtype, 
                                     mask_params.num_context, mask_params.num_target, mask_params.target_group_size)
        golden = self.golden_op_exec(q, k, v, seq_offset, bias, mask, max_seq_len, enable_bias,
                                     mask_params.type, silu_scale, qkv_shape_params.dtype)

        if qkv_shape_params.dtype == torch.bfloat16:
            res = torch.allclose(output, golden, 1e-2)
        elif qkv_shape_params.dtype == torch.float16:
            res = torch.allclose(output, golden, 1e-3)
        else:
            res = torch.allclose(output, golden, 1e-4)
        assert res

    @pytest.mark.parametrize("batch_size", [1, 16])
    @pytest.mark.parametrize("head_num", [2, 4])
    @pytest.mark.parametrize("max_seq_len", [15, 1024])
    @pytest.mark.parametrize("head_dim", [16, 128])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("mask_type", [MaskType.TRIL, MaskType.NONE, MaskType.CUSTOM])
    @pytest.mark.parametrize("silu_scale", [0, 1 / 1024])
    @pytest.mark.parametrize("data_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("num_context", [0, 6])
    @pytest.mark.parametrize("num_target", [30, 512])
    @pytest.mark.parametrize("target_group_size", [1, 3])
    @pytest.mark.skipif(get_chip(), reason="This test case is Skipped for Ascend310P.")
    def test_hstu_dens_forward(self, batch_size, head_num, max_seq_len, head_dim, enable_bias, mask_type, silu_scale,
                               data_type, num_context, num_target, target_group_size):
        qkv_shape_params = QKVShapeParams(dtype=data_type,
                                          batch_size=batch_size,
                                          max_seq_len=max_seq_len,
                                          num_heads=head_num,
                                          attention_dim=head_dim)
        mask_params = MaskParams(type=mask_type,
                                 num_context=num_context,
                                 num_target=num_target,
                                 target_group_size=target_group_size)
        self.execute(qkv_shape_params, mask_params, enable_bias, silu_scale)

    @pytest.mark.parametrize("head_num", [2])
    @pytest.mark.parametrize("max_seq_len", [2570])
    @pytest.mark.parametrize("head_dim", [256])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("mask_type", [MaskType.TRIL, MaskType.NONE, MaskType.CUSTOM])
    @pytest.mark.parametrize("silu_scale", [0, 1 / 1024])
    @pytest.mark.parametrize("data_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("num_context", [0, 6])
    @pytest.mark.parametrize("num_target", [20, 512])
    @pytest.mark.parametrize("target_group_size", [1, 3])
    @pytest.mark.skipif(get_chip(), reason="This test case is Skipped for Ascend310P.")
    def test_hstu_dens_forward_128bs(self, head_num, max_seq_len, head_dim, enable_bias, mask_type, silu_scale,
                                     data_type, num_context, num_target, target_group_size):
        qkv_shape_params = QKVShapeParams(dtype=data_type,
                                          batch_size=128,
                                          max_seq_len=max_seq_len,
                                          num_heads=head_num,
                                          attention_dim=head_dim)
        mask_params = MaskParams(type=mask_type,
                                 num_context=num_context,
                                 num_target=num_target,
                                 target_group_size=target_group_size)
        self.execute(qkv_shape_params, mask_params, enable_bias, silu_scale)

    @pytest.mark.parametrize("head_num", [2])
    @pytest.mark.parametrize("max_seq_len", [16])
    @pytest.mark.parametrize("head_dim", [256])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("mask_type", [MaskType.TRIL, MaskType.NONE, MaskType.CUSTOM])
    @pytest.mark.parametrize("silu_scale", [0, 1 / 1024])
    @pytest.mark.parametrize("data_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("num_context", [0, 3])
    @pytest.mark.parametrize("num_target", [4])
    @pytest.mark.parametrize("target_group_size", [1, 3])
    @pytest.mark.skipif(get_chip(), reason="This test case is Skipped for Ascend310P.")
    def test_hstu_dens_forward_2048bs(self, head_num, max_seq_len, head_dim, enable_bias, mask_type, silu_scale,
                                      data_type, num_context, num_target, target_group_size):
        qkv_shape_params = QKVShapeParams(dtype=data_type,
                                          batch_size=2048,
                                          max_seq_len=max_seq_len,
                                          num_heads=head_num,
                                          attention_dim=head_dim)
        mask_params = MaskParams(type=mask_type,
                                 num_context=num_context,
                                 num_target=num_target,
                                 target_group_size=target_group_size)
        self.execute(qkv_shape_params, mask_params, enable_bias, silu_scale)


class TestHstuNormalDemo:
    @staticmethod
    def golden_op_exec(q, k, v, bias, mask, mask_type, max_seq_len, silu_scale, enable_bias, data_type):
        b, n, num_heads, linear_dim = q.shape
        silu_scale = 1 / max_seq_len if silu_scale == 0 else silu_scale
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        qk_attn = torch.matmul(q, k)

        qk_attn = qk_attn.to(torch.float32)
        bias = bias.to(torch.float32)
        mask = mask.to(torch.float32)
        if enable_bias:
            qk_attn = qk_attn + bias

        qk_attn = F.silu(qk_attn) * silu_scale

        if get_chip():
            mask = mask.repeat(1, num_heads, 1, 1)
            qk_attn = qk_attn * mask
        elif mask_type != MaskType.NONE:
            qk_attn = qk_attn * mask

        v = v.permute(0, 2, 1, 3)

        qk_attn = qk_attn.to(data_type)
        attn_output = torch.matmul(qk_attn, v)
        attn_output = attn_output.permute(0, 2, 1, 3)
        torch.npu.synchronize()
        return attn_output.cpu().to(data_type).reshape(-1)

    @staticmethod
    def custom_op_exec(q, k, v, bias, mask, mask_type, max_seq_len, silu_scale, enable_bias, data_type):
        if enable_bias:
            output = torch.ops.mxrec.hstu_dense(
                q, k, v, mask, bias, mask_type, max_seq_len, silu_scale, "normal"
            )
        else:
            output = torch.ops.mxrec.hstu_dense(
                q, k, v, mask, None, mask_type, max_seq_len, silu_scale, "normal"
            )

        torch.npu.synchronize()
        return output.cpu().to(data_type).reshape(-1)

    def execute(self, batch_size, max_seq_len, head_num, head_dim, enable_bias, mask_type, silu_scale, data_type):
        q, k, v, bias, mask = generate_tensor(batch_size, max_seq_len, head_num, head_dim, data_type, mask_type)

        torch.npu.synchronize()

        output = self.custom_op_exec(q, k, v, bias, mask, mask_type, max_seq_len, silu_scale, enable_bias, data_type)
        golden = self.golden_op_exec(q, k, v, bias, mask, mask_type, max_seq_len, silu_scale, enable_bias, data_type)

        torch.npu.synchronize()

        if data_type == torch.bfloat16:
            res = torch.allclose(output, golden, 1e-2, 1e-2)
        elif data_type == torch.float16:
            res = torch.allclose(output, golden, 1e-3, 1e-3)
        else:
            res = torch.allclose(output, golden, 1e-4, 1e-4)
        assert res

    max_seq_len = [1, 15, 31, 256, 768, 1023, 4095]
    paramFalse = pytest.param(False,
                              marks=pytest.mark.skipif(get_chip(), reason="This test case is Skipped for Ascend310P."))
    paramFp32 = pytest.param(torch.float32,
                             marks=pytest.mark.skipif(get_chip(), reason="This test case is Skipped for Ascend310P."))
    parambF16 = pytest.param(torch.bfloat16,
                             marks=pytest.mark.skipif(get_chip(), reason="This test case is Skipped for Ascend310P."))
    paramsSeqlen = []
    for i in max_seq_len:
        if skip_seq_len(i):
            paramsSeqlen.append(
                pytest.param(i, marks=pytest.mark.skipif(True, reason="This test case is Skipped for Ascend310P.")))
        else:
            paramsSeqlen.append(pytest.param(i))

    @pytest.mark.parametrize("batch_size", [1, 16])
    @pytest.mark.parametrize("head_num", [2, 4])
    @pytest.mark.parametrize("max_seq_len", paramsSeqlen)
    @pytest.mark.parametrize("head_dim", [32, 64])
    @pytest.mark.parametrize("enable_bias", [True, paramFalse])
    @pytest.mark.parametrize("mask_type", [MaskType.TRIL, MaskType.NONE, MaskType.CUSTOM])
    @pytest.mark.parametrize("silu_scale", [1 / 256])
    @pytest.mark.parametrize("data_type", [torch.float16, paramFp32, parambF16])
    def test_hstu_dens_normal(self, batch_size, head_num, max_seq_len, head_dim, enable_bias, mask_type, silu_scale,
                              data_type):
        self.execute(batch_size, max_seq_len, head_num, head_dim, enable_bias, mask_type, silu_scale, data_type)


if __name__ == "__main__":
    min_seq_len, max_seq_len = 1, 2048
    batch_size, head_num, head_dim = 2048, 2, 256
    data_type, mask_type, enable_bias, silu_scale = torch.float16, MaskType.TRIL, True, 1 / 256
    num_context, num_target, target_group_size = 6, 512, 3

    qkv_shape_params = QKVShapeParams(dtype=data_type,
                                    batch_size=batch_size,
                                    min_seq_len=min_seq_len,
                                    max_seq_len=max_seq_len,
                                    num_heads=head_num,
                                    attention_dim=head_dim)
    mask_params = MaskParams(type=mask_type,
                                num_context=num_context,
                                num_target=num_target,
                                target_group_size=target_group_size)
    op = TestHstuJaggedDemo()
    op.execute(qkv_shape_params, mask_params, enable_bias, silu_scale)
